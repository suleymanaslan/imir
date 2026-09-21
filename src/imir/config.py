"""Versioned configuration loaded from an ImIR checkpoint."""

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MapperConfig:
    dim: int = 3584
    hidden_dim: int = 8192
    context_dim: int = 256
    task_dim: int = 64
    # Empty means task-agnostic: one shared learned slot, no label required.
    task_names: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "task_names", tuple(self.task_names))
        for key in ("dim", "hidden_dim", "context_dim", "task_dim"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if any(not isinstance(t, str) or not t.strip() for t in self.task_names):
            raise ValueError("task_names must contain nonempty strings")
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task_names must be unique")


@dataclass(frozen=True)
class ImIRConfig:
    base_model: str
    base_revision: str
    mapper: MapperConfig = field(default_factory=MapperConfig)
    format_version: int = 1
    conditioning: str = "qwen-edit-plus-empty-v1"
    max_pixels: int = 1024 * 1024
    resize_mode: str = "downscale_only"
    instruction_pixels: int = 384 * 384
    num_inference_steps: int = 30
    instruction_scale: float = 1.0

    def __post_init__(self):
        if self.format_version != 1 or self.conditioning != "qwen-edit-plus-empty-v1":
            raise ValueError("Unsupported checkpoint format or conditioning contract")
        if self.resize_mode not in ("downscale_only", "target_area"):
            raise ValueError("resize_mode must be downscale_only or target_area")
        if not self.base_model:
            raise ValueError("base_model is required")
        if not re.fullmatch(r"[0-9a-f]{40}", self.base_revision):
            raise ValueError("base_revision must be an immutable 40-character Hub commit SHA")
        for key in ("max_pixels", "instruction_pixels", "num_inference_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.max_pixels < 256 or self.instruction_pixels < 1024:
            raise ValueError("Pixel budgets must accommodate the model's spatial grids")
        if not math.isfinite(self.instruction_scale):
            raise ValueError("instruction_scale must be finite")

    def to_dict(self):
        data = asdict(self)
        # Preserve compatibility with checkpoints that omit the default resize mode.
        if self.resize_mode == "downscale_only":
            data.pop("resize_mode")
        return data

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text())
        data["mapper"] = MapperConfig(**data["mapper"])
        return cls(**data)
