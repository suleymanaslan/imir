"""Portable inference bundles: JSON + mapper safetensors + Diffusers LoRA."""

import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from .config import ImIRConfig
from .mapper import TokenMapper

CONFIG_NAME = "imir_config.json"
MAPPER_NAME = "mapper.safetensors"
LORA_NAME = "pytorch_lora_weights.safetensors"
BUNDLE_FILES = (CONFIG_NAME, MAPPER_NAME, LORA_NAME)
LORA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_out.0",
    "to_add_out",
    "img_mlp.net.2",
    "img_mod.1",
    "txt_mlp.net.2",
    "txt_mod.1",
)


def resolve_checkpoint(checkpoint, revision=None, local_files_only=False):
    path = Path(checkpoint).expanduser()
    if not path.is_dir():
        if path.is_absolute() or str(checkpoint).startswith((".", "~")):
            raise FileNotFoundError(f"Checkpoint directory not found: {path}")
        path = Path(
            snapshot_download(
                repo_id=str(checkpoint),
                revision=revision,
                allow_patterns=list(BUNDLE_FILES),
                local_files_only=local_files_only,
            )
        )
    for name in BUNDLE_FILES:
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing checkpoint component: {path / name}")
    return path


def load_mapper(path, config):
    mapper = TokenMapper(config.mapper)
    mapper.load_state_dict(load_file(str(Path(path) / MAPPER_NAME), device="cpu"), strict=True)
    return mapper.eval()


def save_checkpoint(
    path, config: ImIRConfig, mapper: TokenMapper, transformer, adapter_name="default"
):
    """Save a PEFT adapter and mapper as a portable inference bundle.

    Destination must not exist. Write to a sibling temporary directory, then rename
    after all components have been serialized successfully.
    """
    from diffusers import QwenImageEditPlusPipeline
    from peft import get_peft_model_state_dict

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing checkpoint: {path}")
    if mapper.config != config.mapper:
        raise ValueError("Mapper architecture does not match the bundle configuration")
    peft_configs = getattr(transformer, "peft_config", {})
    if adapter_name not in peft_configs:
        raise ValueError(f"Transformer has no PEFT adapter {adapter_name!r}")
    peft_config = peft_configs[adapter_name]
    # Standard Diffusers LoRA stores matrices without a separate PEFT config.
    # alpha == rank makes the effective scale unambiguous on reload.
    if (
        peft_config.lora_alpha != peft_config.r
        or peft_config.use_rslora
        or peft_config.use_dora
        or peft_config.bias != "none"
        or getattr(peft_config, "lora_bias", False)
        or peft_config.modules_to_save
        or peft_config.rank_pattern
        or peft_config.alpha_pattern
    ):
        raise ValueError("Export requires ordinary LoRA with alpha=rank and bias='none'")
    if transformer.config.joint_attention_dim != config.mapper.dim:
        raise ValueError("Transformer and mapper conditioning dimensions differ")
    if not transformer.config.zero_cond_t:
        raise ValueError("ImIR v1 requires the 2511 zero_cond_t transformer")
    state = get_peft_model_state_dict(transformer, adapter_name=adapter_name)
    if not state:
        raise ValueError("Adapter contains no weights")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        config.save(staging / CONFIG_NAME)
        save_file(
            {
                k: v.detach().to(device="cpu", dtype=torch.float32).contiguous()
                for k, v in mapper.state_dict().items()
            },
            str(staging / MAPPER_NAME),
        )
        QwenImageEditPlusPipeline.save_lora_weights(
            str(staging),
            transformer_lora_layers=state,
            safe_serialization=True,
            weight_name=LORA_NAME,
        )
        staging.rename(path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
