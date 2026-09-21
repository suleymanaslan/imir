"""ImIR orchestration using pinned Diffusers components.

Latent packing, flow schedule, and normalization follow QwenImageEditPlusPipeline
(Diffusers 0.37.0). The checkpoint controls VAE canvas resizing; conditioning
is a mapped image embedding.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_shift
from PIL import Image

from .checkpoint import CONFIG_NAME, LORA_NAME, load_mapper, resolve_checkpoint
from .conditioning import encode_image
from .config import ImIRConfig
from .images import load_image
from .mapper import TokenMapper
from .resolution import prepare_canvas


@dataclass
class RestorationOutput:
    image: Image.Image
    metadata: dict


class ImIRPipeline:
    def __init__(
        self, backbone: QwenImageEditPlusPipeline, mapper: TokenMapper, config: ImIRConfig
    ):
        if mapper.config != config.mapper:
            raise ValueError("Mapper configuration does not match checkpoint")
        if backbone.transformer.config.joint_attention_dim != config.mapper.dim:
            raise ValueError("Backbone and mapper embedding dimensions differ")
        if not backbone.transformer.config.zero_cond_t:
            raise ValueError("ImIR v1 requires Qwen-Image-Edit-2511 with zero_cond_t=True")
        if backbone.transformer.config.guidance_embeds:
            raise ValueError("Guidance-distilled backbones are not supported")
        if backbone.vae_scale_factor * 2 != 16:
            raise ValueError("ImIR v1 requires a 16-pixel packed latent grid")
        if not isinstance(backbone.scheduler, FlowMatchEulerDiscreteScheduler):
            raise ValueError("ImIR v1 requires FlowMatchEulerDiscreteScheduler")
        self.backbone = backbone
        self.mapper = mapper.eval().float()
        self.config = config
        self.checkpoint_source = None
        self.checkpoint_revision = None
        for model in (backbone.text_encoder, backbone.transformer, backbone.vae):
            if model is not None:
                model.eval()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        revision=None,
        device="cuda",
        torch_dtype=torch.bfloat16,
        cpu_offload=False,
        local_files_only=False,
        base_model_path=None,
    ):
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use a CUDA PyTorch build or device='cpu'")
        if cpu_offload and device.type != "cuda":
            raise ValueError("CPU offload requires a CUDA execution device")
        path = resolve_checkpoint(checkpoint, revision, local_files_only)
        config = ImIRConfig.load(path / CONFIG_NAME)
        mapper = load_mapper(path, config)
        source = str(base_model_path) if base_model_path is not None else config.base_model
        if base_model_path is not None and not Path(base_model_path).is_dir():
            raise FileNotFoundError(f"Local backbone directory not found: {base_model_path}")
        backbone = QwenImageEditPlusPipeline.from_pretrained(
            source,
            revision=None if base_model_path is not None else config.base_revision,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        backbone.load_lora_weights(str(path), weight_name=LORA_NAME, adapter_name="imir")
        if cpu_offload:
            backbone.enable_model_cpu_offload(device=str(device))
        else:
            backbone.to(device)
        instance = cls(backbone, mapper, config)
        instance.checkpoint_source = str(checkpoint)
        instance.checkpoint_revision = path.name if path.parent.name == "snapshots" else revision
        return instance

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input image as a sequence of vision-language tokens.

        Includes the stock edit-plus template with an empty user instruction.
        Mask is always bool [1,L], True for valid tokens; never truncate tokens.
        """
        embeddings, mask = encode_image(self.backbone, image, self.config.instruction_pixels)
        if embeddings.shape[-1] != self.config.mapper.dim:
            raise ValueError("Encoder output dimension does not match mapper")
        return embeddings, mask.bool()

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image | str | Path,
        *,
        task=None,
        instruction_scale=None,
        num_inference_steps=None,
        seed=0,
        restore_original_size=True,
    ):
        task_id = self.mapper.task_id(task)
        steps = (
            self.config.num_inference_steps if num_inference_steps is None else num_inference_steps
        )
        scale = self.config.instruction_scale if instruction_scale is None else instruction_scale
        if type(steps) is not int or steps < 1:
            raise ValueError("num_inference_steps must be a positive integer")
        if not math.isfinite(scale):
            raise ValueError("instruction_scale must be finite")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        source = load_image(image)
        canvas = prepare_canvas(source, self.config.max_pixels, self.config.resize_mode)
        width, height = canvas.size
        pipe = self.backbone
        device = pipe._execution_device
        # Local CPU generator: reproducible seeds without changing global RNG state.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        try:
            embeddings, mask = self.encode_image(canvas)
            self.mapper.to(device=embeddings.device, dtype=torch.float32)
            task_ids = torch.tensor([task_id], device=embeddings.device, dtype=torch.long)
            mapped = self.mapper.condition(embeddings.float(), mask, task_ids, scale)
            mapped = mapped.to(dtype=pipe.transformer.dtype)
            pixels = pipe.image_processor.preprocess(canvas, height, width).unsqueeze(2)
            latents, structure = pipe.prepare_latents(
                pixels,
                1,
                pipe.transformer.config.in_channels // 4,
                height,
                width,
                mapped.dtype,
                device,
                generator,
            )
            # Both generated and conditioning images use exactly the same frame.
            grid = (1, height // 16, width // 16)
            img_shapes = [[grid, grid]]
            # Fresh scheduler per call avoids carrying mutable sampler state across images.
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            mu = calculate_shift(
                latents.shape[1],
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )
            scheduler.set_timesteps(
                sigmas=np.linspace(1.0, 1.0 / steps, steps),
                device=device,
                mu=mu,
            )
            scheduler.set_begin_index(0)
            for timestep in pipe.progress_bar(scheduler.timesteps):
                model_input = torch.cat((latents, structure), dim=1)
                prediction = pipe.transformer(
                    hidden_states=model_input,
                    timestep=timestep.expand(1).to(latents.dtype) / 1000,
                    guidance=None,
                    encoder_hidden_states=mapped,
                    encoder_hidden_states_mask=mask,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0][:, : latents.shape[1]]
                latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
            unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
            unpacked = unpacked.to(dtype=pipe.vae.dtype)
            mean = torch.tensor(
                pipe.vae.config.latents_mean, device=device, dtype=unpacked.dtype
            ).view(1, -1, 1, 1, 1)
            std = torch.tensor(
                pipe.vae.config.latents_std, device=device, dtype=unpacked.dtype
            ).view(1, -1, 1, 1, 1)
            decoded = pipe.vae.decode(unpacked * std + mean, return_dict=False)[0][:, :, 0]
            restored = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
            if restore_original_size and restored.size != source.size:
                restored = restored.resize(source.size, Image.Resampling.LANCZOS)
            metadata = {
                "format_version": self.config.format_version,
                "checkpoint": self.checkpoint_source,
                "checkpoint_revision": self.checkpoint_revision,
                "base_model": self.config.base_model,
                "base_revision": self.config.base_revision,
                "conditioning": self.config.conditioning,
                "input_size": list(source.size),
                "generation_size": [width, height],
                "output_size": list(restored.size),
                "seed": seed,
                "num_inference_steps": steps,
                "instruction_scale": scale,
                "task": task,
                "max_pixels": self.config.max_pixels,
                "resize_mode": self.config.resize_mode,
                "instruction_pixels": self.config.instruction_pixels,
                "dtype": str(pipe.transformer.dtype),
                "device": str(device),
                "diffusers_version": "0.37.0",
                "torch_version": str(torch.__version__),
                "true_cfg_scale": 1.0,
            }
            return RestorationOutput(restored, metadata)
        finally:
            pipe.maybe_free_model_hooks()
