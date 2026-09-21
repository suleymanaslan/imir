from dataclasses import replace

import numpy as np
import pytest
import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
)
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image

from imir.checkpoint import LORA_TARGET_MODULES, load_mapper, save_checkpoint
from imir.config import ImIRConfig, MapperConfig
from imir.mapper import TokenMapper
from imir.pipeline import ImIRPipeline


@pytest.fixture
def components():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    transformer = QwenImageTransformer2DModel(
        in_channels=16,
        out_channels=4,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=8,
        axes_dims_rope=(4, 6, 6),
        zero_cond_t=True,
    )
    vae = AutoencoderKLQwenImage(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        latents_mean=[0.1] * 4,
        latents_std=[1.5] * 4,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    backbone = QwenImageEditPlusPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        text_encoder=None,
        tokenizer=None,
        processor=None,
    )
    backbone.set_progress_bar_config(disable=True)
    config = ImIRConfig(
        base_model="Qwen/Qwen-Image-Edit-2511",
        base_revision="a" * 40,
        mapper=MapperConfig(dim=8, hidden_dim=16, context_dim=4, task_dim=2),
        max_pixels=1024,
        num_inference_steps=2,
    )
    mapper = TokenMapper(config.mapper)
    return backbone, mapper, config


def install_encoder_stub(monkeypatch, backbone):
    # Only the heavyweight VLM is substituted. Sampling uses real Diffusers
    # transformer, VAE, latent helpers, LoRA loader, and flow scheduler.
    calls = []
    embeddings = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8) / 40

    def encode_prompt(**kwargs):
        calls.append(kwargs)
        return embeddings.clone(), None

    monkeypatch.setattr(backbone, "encode_prompt", encode_prompt)
    return calls


def test_real_sampling_seed_resolution_and_conditioning(components, monkeypatch):
    backbone, mapper, config = components
    calls = install_encoder_stub(monkeypatch, backbone)
    pipe = ImIRPipeline(backbone, mapper, config)
    image = Image.new("RGB", (37, 35), (20, 30, 40))
    seen = []
    hook = backbone.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs),
        with_kwargs=True,
    )
    rng_before = torch.get_rng_state().clone()
    first = pipe(image, seed=123)
    second = pipe(image, seed=123)
    third = pipe(image, seed=321, restore_original_size=False)
    hook.remove()
    assert torch.equal(rng_before, torch.get_rng_state())
    assert np.array_equal(first.image, second.image)
    assert first.image.size == (37, 35)
    assert third.image.size == (32, 16)
    assert first.metadata["generation_size"] == [32, 16]
    assert not np.array_equal(np.asarray(first.image.resize(third.image.size)), third.image)
    assert calls[0]["prompt"] == ""
    assert len(calls[0]["image"]) == 1
    assert seen[0]["img_shapes"] == [[(1, 1, 2), (1, 1, 2)]]
    assert seen[0]["hidden_states"].shape == (1, 4, 16)
    assert seen[0]["encoder_hidden_states_mask"].dtype == torch.bool
    assert torch.isfinite(seen[0]["encoder_hidden_states"]).all()


def test_mapper_changes_actual_transformer_conditioning(components, monkeypatch):
    backbone, mapper, config = components
    install_encoder_stub(monkeypatch, backbone)
    torch.nn.init.normal_(mapper.output_projection.weight)
    pipe = ImIRPipeline(backbone, mapper, config)
    image = Image.new("RGB", (32, 32))
    identity = pipe(image, instruction_scale=0, seed=0)
    mapped = pipe(image, instruction_scale=2, seed=0)
    assert not np.array_equal(identity.image, mapped.image)


def test_checkpoint_roundtrip_with_real_peft_loader(components, monkeypatch, tmp_path):
    backbone, mapper, config = components
    backbone.transformer.add_adapter(
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=list(LORA_TARGET_MODULES),
        )
    )
    for name, parameter in backbone.transformer.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.1)
    expected = get_peft_model_state_dict(backbone.transformer)
    destination = tmp_path / "checkpoint"
    save_checkpoint(destination, config, mapper, backbone.transformer)
    loaded_mapper = load_mapper(destination, config)
    for name, value in mapper.state_dict().items():
        assert torch.equal(value, loaded_mapper.state_dict()[name])
    backbone.unload_lora_weights()
    calls = []

    def load_backbone(source, **kwargs):
        calls.append((source, kwargs))
        return backbone

    monkeypatch.setattr(QwenImageEditPlusPipeline, "from_pretrained", load_backbone)
    pipe = ImIRPipeline.from_pretrained(destination, device="cpu", torch_dtype=torch.float32)
    actual = get_peft_model_state_dict(pipe.backbone.transformer, adapter_name="imir")
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[k], actual[k]) for k in expected)
    assert calls[0][1]["revision"] == config.base_revision
    install_encoder_stub(monkeypatch, backbone)
    assert pipe(Image.new("RGB", (32, 32))).image.size == (32, 32)
    with pytest.raises(FileExistsError):
        save_checkpoint(destination, config, mapper, backbone.transformer, adapter_name="imir")


def test_export_rejects_ambiguous_lora_scaling(components, tmp_path):
    backbone, mapper, config = components
    backbone.transformer.add_adapter(LoraConfig(r=2, lora_alpha=4, target_modules=["to_q"]))
    with pytest.raises(ValueError, match="alpha=rank"):
        save_checkpoint(tmp_path / "bad", config, mapper, backbone.transformer)
    assert not (tmp_path / "bad").exists()


def test_task_aware_requires_label_before_encoding(components, monkeypatch):
    backbone, _, config = components
    config = replace(config, mapper=replace(config.mapper, task_names=("denoising", "dehazing")))
    calls = install_encoder_stub(monkeypatch, backbone)
    pipe = ImIRPipeline(backbone, TokenMapper(config.mapper), config)
    with pytest.raises(ValueError, match="Choose a task"):
        pipe(Image.new("RGB", (32, 32)))
    assert not calls
    assert pipe(Image.new("RGB", (32, 32)), task="denoising").metadata["task"] == "denoising"


def test_real_vlm_encoding_with_synthetic_processor(components):
    from transformers import BatchFeature, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    backbone, mapper, config = components
    vlm_config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rope_scaling": {"rope_type": "default", "mrope_section": [0, 1, 1]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": 8,
            "fullatt_block_indexes": [0],
        },
        image_token_id=3,
        video_token_id=6,
        vision_start_token_id=4,
        vision_end_token_id=5,
    )
    backbone.text_encoder = Qwen2_5_VLForConditionalGeneration(vlm_config).eval()

    class SyntheticProcessor:
        def __call__(self, text, images, padding, return_tensors):
            assert "Picture 1:" in text[0]
            assert "<|vision_end|><|im_end|>" in text[0]
            assert len(images) == 1 and images[0].mode == "RGB"
            ids = torch.tensor([[1] * 63 + [4, 3, 5, 1, 1]])
            return BatchFeature(
                {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "pixel_values": torch.zeros(4, 3 * 2 * 14 * 14),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                }
            )

    backbone.processor = SyntheticProcessor()
    pipe = ImIRPipeline(backbone, mapper, config)
    embeddings, mask = pipe.encode_image(Image.new("RGB", (32, 32)))
    assert embeddings.shape == (1, 4, 8)
    assert mask.shape == (1, 4) and mask.all()
    assert torch.isfinite(embeddings).all() and not embeddings.requires_grad
    assert pipe(Image.new("RGB", (32, 32)), seed=42).image.size == (32, 32)


def test_cli_writes_png_and_provenance(components, monkeypatch, tmp_path):
    import json

    from imir.cli import main

    backbone, mapper, config = components
    install_encoder_stub(monkeypatch, backbone)
    pipe = ImIRPipeline(backbone, mapper, config)
    monkeypatch.setattr(ImIRPipeline, "from_pretrained", lambda *args, **kwargs: pipe)
    source = tmp_path / "input.png"
    Image.new("RGB", (32, 32)).save(source)
    output = tmp_path / "output.png"
    main(
        [
            "--checkpoint",
            "./checkpoint",
            "--input",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--seed",
            "77",
        ]
    )
    with Image.open(output) as image:
        assert image.size == (32, 32) and image.format == "PNG"
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["seed"] == 77
    assert metadata["input_file"] == str(source)
    assert metadata["base_revision"] == config.base_revision


def test_target_area_sampling_uses_enlarged_frame_and_restores_original(components, monkeypatch):
    backbone, mapper, config = components
    install_encoder_stub(monkeypatch, backbone)
    config = replace(config, max_pixels=4096, resize_mode="target_area")
    pipe = ImIRPipeline(backbone, mapper, config)
    image = Image.new("RGB", (32, 32), (20, 30, 40))
    seen = []
    hook = backbone.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["img_shapes"]), with_kwargs=True
    )
    result = pipe(image, seed=42)
    hook.remove()
    assert result.image.size == image.size
    assert result.metadata["generation_size"] == [64, 64]
    assert result.metadata["resize_mode"] == "target_area"
    assert seen[0] == [[(1, 4, 4), (1, 4, 4)]]
