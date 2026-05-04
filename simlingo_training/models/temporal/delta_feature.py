from typing import Optional

import math
import torch
from torch import nn
import torch.nn.functional as F


class TemporalDeltaFeatureEncoder(nn.Module):
    """
    DeltaFlow-inspired RGB feature adapter.

    Instead of treating the temporal history as a separate sequence, this module
    first collapses current-vs-past feature differences into one shared motion
    feature map. A lightweight pooling adapter then converts that motion map into
    the fixed number of temporal tokens expected by SimLingo.
    """

    uses_current_frame = True

    def __init__(
        self,
        hidden_size: int,
        max_history_frames: int,
        num_queries: int = 16,
        dropout: float = 0.1,
        gate_init: float = 0.0,
        include_absolute_delta: bool = True,
        delta_decay: float = 0.9,
        **_: Optional[object],
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.max_history_frames = max(1, max_history_frames)
        self.include_absolute_delta = include_absolute_delta
        self.delta_decay = delta_decay

        delta_input_size = hidden_size * (2 if include_absolute_delta else 1)
        self.delta_projection = nn.Sequential(
            nn.LayerNorm(delta_input_size),
            nn.Linear(delta_input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.motion_norm = nn.LayerNorm(hidden_size)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float))

    def get_time_weights(self, num_past_frames: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Past frames are ordered oldest -> newest. With decay < 1, older deltas
        # fade slightly, like a feature-space motion trail.
        exponents = torch.arange(num_past_frames - 1, -1, -1, device=device, dtype=dtype)
        return torch.pow(torch.as_tensor(self.delta_decay, device=device, dtype=dtype), exponents)

    def pool_motion_features(self, motion_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, hidden_size = motion_features.shape
        input_side = math.isqrt(num_tokens)
        output_side = math.isqrt(self.num_queries)
        if input_side * input_side == num_tokens and output_side * output_side == self.num_queries:
            motion_grid = motion_features.transpose(1, 2).reshape(batch_size, hidden_size, input_side, input_side)
            pooled_grid = F.adaptive_avg_pool2d(motion_grid, (output_side, output_side))
            return pooled_grid.flatten(2).transpose(1, 2)

        return F.adaptive_avg_pool1d(
            motion_features.transpose(1, 2),
            self.num_queries,
        ).transpose(1, 2)

    def forward(
        self,
        past_frame_tokens: torch.Tensor,
        current_frame_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            past_frame_tokens: [B, T_past, P, D], oldest to newest
            current_frame_tokens: [B, P, D] or [B, 1, P, D]

        Returns:
            [B, Q, D] compact motion tokens built from visual feature deltas.
        """
        batch_size = past_frame_tokens.size(0)
        if past_frame_tokens.size(1) == 0:
            return past_frame_tokens.new_zeros(batch_size, self.num_queries, self.hidden_size)
        if current_frame_tokens is None:
            raise ValueError("TemporalDeltaFeatureEncoder requires current_frame_tokens.")
        if current_frame_tokens.dim() == 4:
            current_frame_tokens = current_frame_tokens.squeeze(1)

        num_past_frames = past_frame_tokens.size(1)
        if num_past_frames > self.max_history_frames:
            raise ValueError(
                f"Received {num_past_frames} past frames, but TemporalDeltaFeatureEncoder was "
                f"initialised for at most {self.max_history_frames}."
            )

        current_frame_tokens = current_frame_tokens.unsqueeze(1)
        delta_tokens = current_frame_tokens - past_frame_tokens
        if self.include_absolute_delta:
            delta_tokens = torch.cat((delta_tokens, delta_tokens.abs()), dim=-1)

        time_weights = self.get_time_weights(
            num_past_frames,
            device=delta_tokens.device,
            dtype=delta_tokens.dtype,
        ).view(1, num_past_frames, 1, 1)
        shared_delta_features = (delta_tokens * time_weights).sum(dim=1) / time_weights.sum()
        motion_features = self.delta_projection(shared_delta_features)
        motion_features = self.motion_norm(motion_features)

        pooled_motion = self.pool_motion_features(motion_features)
        temporal_tokens = self.output_norm(self.token_projection(pooled_motion))

        return torch.sigmoid(self.gate) * temporal_tokens
