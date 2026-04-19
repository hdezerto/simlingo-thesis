from typing import Optional

import torch
from torch import nn


class TemporalQFormerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
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
        norm_queries = self.norm_self(queries)
        self_attended, _ = self.self_attn(norm_queries, norm_queries, norm_queries, need_weights=False)
        queries = queries + self.dropout(self_attended)

        norm_queries = self.norm_cross(queries)
        cross_attended, _ = self.cross_attn(norm_queries, memory, memory, need_weights=False)
        queries = queries + self.dropout(cross_attended)

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
        gate_init: float = -2.0,
        **_: Optional[object],
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.max_history_frames = max(1, max_history_frames)

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)
        self.temporal_position_embeddings = nn.Parameter(
            torch.randn(1, self.max_history_frames, 1, hidden_size) * 0.02
        )
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList(
            [TemporalQFormerLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float))

    def forward(self, past_frame_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            past_frame_tokens: [B, T_past, P, D]

        Returns:
            [B, Q, D] temporal memory tokens
        """
        batch_size = past_frame_tokens.size(0)
        if past_frame_tokens.size(1) == 0:
            return past_frame_tokens.new_zeros(batch_size, self.num_queries, self.hidden_size)

        num_past_frames = past_frame_tokens.size(1)
        if num_past_frames > self.max_history_frames:
            raise ValueError(
                f"Received {num_past_frames} past frames, but TemporalQFormer was initialised for "
                f"at most {self.max_history_frames}."
            )

        temporal_pos = self.temporal_position_embeddings[:, :num_past_frames]
        memory = past_frame_tokens + temporal_pos
        memory = memory.reshape(batch_size, -1, self.hidden_size)
        memory = self.memory_norm(memory)

        queries = self.query_tokens.expand(batch_size, -1, -1)
        for layer in self.layers:
            queries = layer(queries, memory)

        return torch.sigmoid(self.gate) * self.output_norm(queries)
