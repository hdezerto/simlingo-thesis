from typing import Optional

import torch
from torch import nn

# Compatibility shim for checkpoints/configs created before the delta encoder
# moved to delta_feature.py. New configs should target delta_feature directly.
from .delta_feature import TemporalDeltaFeatureEncoder


class TemporalQFormerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        # Queries first exchange information with each other so they can specialize
        # before reading from the visual memory.
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Queries then attend to the flattened past-frame memory and pull out the
        # temporal cues they need.
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm_self = nn.LayerNorm(hidden_size)
        self.norm_cross = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # queries: [B, Q, D], memory: [B, T_past * P, D]
        norm_queries = self.norm_self(queries)
        self_attended, _ = self.self_attn(norm_queries, norm_queries, norm_queries, need_weights=False)
        # Residual update: refine each query without discarding its previous state.
        queries = queries + self.dropout(self_attended)

        norm_queries = self.norm_cross(queries)
        cross_attended, _ = self.cross_attn(norm_queries, memory, memory, need_weights=False)
        # Cross-attention lets each learned query read from all past-frame tokens.
        queries = queries + self.dropout(cross_attended)

        # The MLP mixes the attended features within each query token.
        return queries + self.dropout(self.mlp(queries))


class TemporalQFormer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_history_frames: int,
        num_queries: int = 8,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_enabled: bool = True,
        gate_init: float = -2.0,
        use_current_delta: bool = False,
        include_past_features: bool = True,
        include_absolute_delta: bool = True,
        **_: Optional[object],
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.max_history_frames = max(1, max_history_frames)
        self.gate_enabled = gate_enabled
        self.use_current_delta = use_current_delta
        self.uses_current_frame = use_current_delta
        self.include_past_features = include_past_features
        self.include_absolute_delta = include_absolute_delta

        qformer_input_dim = hidden_size
        if self.use_current_delta:
            num_input_parts = 1  # signed current-vs-past feature delta
            if self.include_past_features:
                num_input_parts += 1
            if self.include_absolute_delta:
                num_input_parts += 1
            qformer_input_dim = hidden_size * num_input_parts
        self.input_projection = (
            nn.Linear(qformer_input_dim, hidden_size)
            if qformer_input_dim != hidden_size
            else nn.Identity()
        )

        # Learned summary slots that compress all past frames into a small token set.
        # They are expanded per batch item in forward().
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)
        # One learned embedding per temporal position. The second dimension indexes
        # past frames from oldest to newest; the singleton token dimension is
        # broadcast across all visual tokens from that frame.
        self.temporal_position_embeddings = nn.Parameter(
            torch.randn(1, self.max_history_frames, 1, hidden_size) * 0.02
        )
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList(
            [TemporalQFormerLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        if self.gate_enabled:
            self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float))

    def forward(
        self,
        past_frame_tokens: torch.Tensor,
        current_frame_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            past_frame_tokens: [B, T_past, P, D]
                B: batch size
                T_past: number of past frames (oldest to newest)
                P: number of visual tokens per frame
                D: hidden size
            current_frame_tokens: [B, P, D], required when use_current_delta=True

        Returns:
            [B, Q, D] temporal memory tokens, where Q is num_queries
        """
        batch_size = past_frame_tokens.size(0)
        if past_frame_tokens.size(1) == 0:
            # At sequence start there may be no past frames yet; return zeroed
            # temporal tokens so the rest of the model can keep the same shape.
            return past_frame_tokens.new_zeros(batch_size, self.num_queries, self.hidden_size)

        num_past_frames = past_frame_tokens.size(1)
        if num_past_frames > self.max_history_frames:
            raise ValueError(
                f"Received {num_past_frames} past frames, but TemporalQFormer was initialised for "
                f"at most {self.max_history_frames}."
            )

        if self.use_current_delta:
            if current_frame_tokens is None:
                raise ValueError("current_frame_tokens is required when use_current_delta=True.")
            current_tokens = current_frame_tokens.unsqueeze(1)
            signed_delta = current_tokens - past_frame_tokens
            memory_parts = [signed_delta]
            if self.include_past_features:
                memory_parts.insert(0, past_frame_tokens)
            if self.include_absolute_delta:
                memory_parts.append(signed_delta.abs())
            memory = torch.cat(memory_parts, dim=-1)
            memory = self.input_projection(memory)
        else:
            memory = past_frame_tokens

        # Slice the learned temporal positions for the frames that are actually
        # present in this batch item.
        temporal_pos = self.temporal_position_embeddings[:, :num_past_frames]
        # Add temporal identity before flattening so tokens still carry which past
        # frame they came from after the per-frame structure is removed.
        memory = memory + temporal_pos
        # Merge the per-frame token grids into one long memory sequence:
        # [B, T_past, P, D] -> [B, T_past * P, D]
        memory = memory.reshape(batch_size, -1, self.hidden_size)
        memory = self.memory_norm(memory)

        # Give each batch item its own copy of the learned query tokens.
        queries = self.query_tokens.expand(batch_size, -1, -1)
        for layer in self.layers:
            queries = layer(queries, memory)

        queries = self.output_norm(queries)
        if not self.gate_enabled:
            return queries

        # The gate can start small so the temporal branch learns to help gradually
        # instead of overwhelming the baseline image path early in training.
        return torch.sigmoid(self.gate) * queries
