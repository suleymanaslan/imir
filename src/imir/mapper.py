"""Per-token residual MLP with FiLM from task and masked global image context."""

import math

import torch
from torch import nn

from .config import MapperConfig


class TokenMapper(nn.Module):
    def __init__(self, config: MapperConfig):
        super().__init__()
        self.config = config
        self.context = nn.Sequential(
            nn.Linear(config.dim, config.context_dim),
            nn.GELU(),
            nn.Linear(config.context_dim, config.context_dim),
        )
        self.task_embedding = nn.Embedding(max(1, len(config.task_names)), config.task_dim)
        self.norm = nn.LayerNorm(config.dim)
        self.input_projection = nn.Linear(config.dim, config.hidden_dim)
        self.activation = nn.GELU()
        self.film = nn.Linear(config.context_dim + config.task_dim, 2 * config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def task_id(self, task: str | None) -> int:
        names = self.config.task_names
        if not names:
            if task is not None:
                raise ValueError("This checkpoint is task-agnostic; omit task")
            return 0
        if task not in names:
            raise ValueError(f"Choose a task from {names}; got {task!r}")
        return names.index(task)

    def forward(self, embeddings, valid_mask, task_ids=None):
        """Return residuals [B,L,D]. Boolean mask uses True for VALID tokens."""
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.config.dim:
            raise ValueError(f"Expected embeddings [B,L,{self.config.dim}]")
        if valid_mask.shape != embeddings.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean [B,L], True for valid tokens")
        if not valid_mask.any(dim=1).all():
            raise ValueError("Every image must have at least one valid token")
        batch = embeddings.shape[0]
        if task_ids is None:
            if self.config.task_names:
                raise ValueError("Task-aware mapper requires task_ids")
            task_ids = torch.zeros(batch, dtype=torch.long, device=embeddings.device)
        if task_ids.shape != (batch,) or task_ids.dtype != torch.long:
            raise ValueError("task_ids must be int64 [B]")
        if ((task_ids < 0) | (task_ids >= self.task_embedding.num_embeddings)).any():
            raise ValueError("task_ids out of range")
        # Mask before pooling/MLP: even NaNs in padding cannot contaminate valid tokens.
        x = embeddings.masked_fill(~valid_mask.unsqueeze(-1), 0)
        pooled = x.float().sum(1) / valid_mask.sum(1, keepdim=True)
        context = self.context(pooled.to(x.dtype))
        task = self.task_embedding(task_ids)
        gamma, beta = self.film(torch.cat((context, task), dim=-1)).chunk(2, dim=-1)
        hidden = self.activation(self.input_projection(self.norm(x)))
        hidden = (1 + gamma[:, None]) * hidden + beta[:, None]
        return self.output_projection(hidden).masked_fill(~valid_mask.unsqueeze(-1), 0)

    def condition(self, embeddings, valid_mask, task_ids=None, scale=1.0):
        """Return e_y + scale * residual, with zero padding. Keeps gradients."""
        if not math.isfinite(scale):
            raise ValueError("scale must be finite")
        # Keep frozen conditioning arithmetic in its explicit parameter dtype even
        # when the diffusion sampler runs under an outer autocast context.
        with torch.autocast(embeddings.device.type, enabled=False):
            clean = embeddings.masked_fill(~valid_mask.unsqueeze(-1), 0)
            return clean + scale * self(embeddings, valid_mask, task_ids)
