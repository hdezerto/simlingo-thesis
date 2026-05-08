from typing import Optional, Tuple

import math
import torch
import warnings
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
        gate_enabled: bool = True,
        gate_init: float = 0.0,
        include_absolute_delta: bool = True,
        delta_decay: float = 0.9,
        spatial_pooling: bool = True,
        **_: Optional[object],
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.max_history_frames = max(1, max_history_frames)
        self.gate_enabled = gate_enabled
        self.include_absolute_delta = include_absolute_delta
        self.delta_decay = delta_decay
        self.spatial_pooling = spatial_pooling
        self._warned_spatial_pooling_fallback = False

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
        if self.gate_enabled:
            self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float))

    def get_time_weights(self, num_past_frames: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Past frames are ordered oldest -> newest. With decay < 1, older deltas
        # fade slightly, like a feature-space motion trail.
        exponents = torch.arange(num_past_frames - 1, -1, -1, device=device, dtype=dtype)
        return torch.pow(torch.as_tensor(self.delta_decay, device=device, dtype=dtype), exponents)

    def pool_motion_features(self, motion_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, hidden_size = motion_features.shape
        if self.spatial_pooling:
            pooled_motion = self._pool_spatial_token_blocks(motion_features)
            if pooled_motion is not None:
                return pooled_motion
            self._warn_spatial_pooling_fallback(num_tokens)

        return F.adaptive_avg_pool1d(
            motion_features.transpose(1, 2),
            self.num_queries,
        ).transpose(1, 2)

    def _warn_spatial_pooling_fallback(self, num_tokens: int) -> None:
        if self._warned_spatial_pooling_fallback:
            return
        warnings.warn(
            "TemporalDeltaFeatureEncoder spatial_pooling=True, but the visual token "
            f"layout could not be inferred from {num_tokens} tokens; falling back to 1D pooling.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._warned_spatial_pooling_fallback = True

    def _pool_spatial_token_blocks(self, motion_features: torch.Tensor) -> Optional[torch.Tensor]:
        batch_size, num_tokens, hidden_size = motion_features.shape
        token_block_shape = self._infer_square_token_blocks(num_tokens)
        if token_block_shape is None:
            return None

        num_blocks, block_side = token_block_shape
        if self.num_queries % num_blocks == 0:
            per_block_tokens = self.num_queries // num_blocks
            block_out_h, block_out_w = self._factor_pair_closest_to_aspect(
                per_block_tokens,
                target_aspect=1.0,
            )
            blocks = motion_features.reshape(batch_size, num_blocks, block_side, block_side, hidden_size)
            blocks = blocks.permute(0, 1, 4, 2, 3).reshape(
                batch_size * num_blocks,
                hidden_size,
                block_side,
                block_side,
            )
            pooled_blocks = F.adaptive_avg_pool2d(blocks, (block_out_h, block_out_w))
            pooled_blocks = pooled_blocks.reshape(
                batch_size,
                num_blocks,
                hidden_size,
                block_out_h,
                block_out_w,
            )
            # InternVL dynamic preprocessing stores the front image blocks in
            # row-major order. SimLingo uses two horizontal blocks, so keeping
            # the block axis before the per-block width preserves left/right
            # locality instead of flattening all 512 tokens into a 1D sequence.
            return pooled_blocks.permute(0, 3, 1, 4, 2).reshape(
                batch_size,
                self.num_queries,
                hidden_size,
            )

        grid = motion_features.reshape(batch_size, num_blocks, block_side, block_side, hidden_size)
        grid = grid.permute(0, 4, 2, 1, 3).reshape(
            batch_size,
            hidden_size,
            block_side,
            num_blocks * block_side,
        )
        out_h, out_w = self._factor_pair_closest_to_aspect(
            self.num_queries,
            target_aspect=float(num_blocks),
        )
        pooled_grid = F.adaptive_avg_pool2d(grid, (out_h, out_w))
        return pooled_grid.flatten(2).transpose(1, 2)

    @staticmethod
    def _infer_square_token_blocks(num_tokens: int) -> Optional[Tuple[int, int]]:
        for block_side in range(math.isqrt(num_tokens), 1, -1):
            block_tokens = block_side * block_side
            if num_tokens % block_tokens == 0:
                return num_tokens // block_tokens, block_side
        return None

    @staticmethod
    def _factor_pair_closest_to_aspect(num_cells: int, target_aspect: float) -> Tuple[int, int]:
        best_h, best_w = 1, num_cells
        best_score = float("inf")
        target_aspect = max(target_aspect, 1e-6)
        for h in range(1, math.isqrt(num_cells) + 1):
            if num_cells % h != 0:
                continue
            for candidate_h, candidate_w in ((h, num_cells // h), (num_cells // h, h)):
                aspect = candidate_w / candidate_h
                score = abs(math.log(aspect / target_aspect))
                if score < best_score - 1e-12 or (
                    abs(score - best_score) <= 1e-12 and candidate_w > best_w
                ):
                    best_score = score
                    best_h = candidate_h
                    best_w = candidate_w
        return best_h, best_w

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
        if not self.gate_enabled:
            return temporal_tokens

        return torch.sigmoid(self.gate) * temporal_tokens
