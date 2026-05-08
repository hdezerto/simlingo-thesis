import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


class TemporalSignalDiagnostic:
    """Writes temporal feature/token diagnostics during selected render runs."""

    def __init__(
        self,
        model,
        device: torch.device,
        output_root: Path,
        history_stride: int,
        temporal_inference_stride: int,
        num_image_tokens_per_patch: int,
        stride: int = 5,
        save_heatmaps: bool = True,
    ):
        self.model = model
        self.device = device
        self.temporal_encoder = getattr(model, "temporal_encoder", None)
        self.history_stride = history_stride
        self.temporal_inference_stride = temporal_inference_stride
        self.num_image_tokens_per_patch = num_image_tokens_per_patch
        self.stride = max(1, stride)
        self.save_heatmaps = save_heatmaps

        self.stats_dir = output_root / "stats"
        self.heatmap_dir = output_root / "heatmaps"
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        if self.save_heatmaps:
            self.heatmap_dir.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        encoder_name = (
            self.temporal_encoder.__class__.__name__
            if self.temporal_encoder is not None
            else "None"
        )
        return (
            "Temporal signal diagnostic enabled "
            f"(encoder={encoder_name}, stride={self.stride}, heatmaps={self.save_heatmaps})"
        )

    def write_if_due(self, frame_features: Optional[torch.Tensor], frame_index: int) -> None:
        if frame_index % self.stride != 0:
            return
        self.write(frame_features, frame_index)

    def write(self, frame_features: Optional[torch.Tensor], frame_index: int) -> None:
        if self.temporal_encoder is None or frame_features is None:
            return

        frame_features = frame_features.detach()
        if frame_features.dim() != 4 or frame_features.size(1) < 2:
            return

        past_frame_features = frame_features[:, :-1]
        current_frame_features = frame_features[:, -1]
        repeated_current_features = current_frame_features.unsqueeze(1).expand_as(past_frame_features)

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            real_tokens = self._call_temporal_encoder(past_frame_features, current_frame_features)
            repeat_tokens = self._call_temporal_encoder(repeated_current_features, current_frame_features)
        zero_tokens = torch.zeros_like(real_tokens)

        deltas = current_frame_features.unsqueeze(1).float() - past_frame_features.float()
        delta_norms = deltas.norm(dim=-1)
        per_history_mean = delta_norms.mean(dim=-1)[0]
        per_history_max = delta_norms.max(dim=-1).values[0]
        weighted_delta_heatmaps = self._compute_weighted_delta_heatmaps(
            past_frame_features,
            current_frame_features,
        )

        stats = {
            "frame_index": int(frame_index),
            "temporal_encoder": self.temporal_encoder.__class__.__name__,
            "history_stride": int(self.history_stride),
            "temporal_inference_stride": int(self.temporal_inference_stride),
            "feature_shape": [int(x) for x in frame_features.shape],
            "gate": self._gate_info(),
            "input_delta_norm": {
                "mean": float(delta_norms.mean().cpu()),
                "max": float(delta_norms.max().cpu()),
                "per_history_frame_mean": [float(x.cpu()) for x in per_history_mean],
                "per_history_frame_max": [float(x.cpu()) for x in per_history_max],
            },
            "diagnostic_heatmaps": self._heatmap_stats(weighted_delta_heatmaps),
            "temporal_token_norm": {
                "real": self._token_norm_stats(real_tokens),
                "repeat_current": self._token_norm_stats(repeat_tokens),
                "zero": self._token_norm_stats(zero_tokens),
            },
            "temporal_token_distance": {
                "real_vs_repeat_current_l2": self._mean_token_l2(real_tokens, repeat_tokens),
                "real_vs_zero_l2": self._mean_token_l2(real_tokens, zero_tokens),
                "repeat_current_vs_zero_l2": self._mean_token_l2(repeat_tokens, zero_tokens),
                "real_vs_repeat_current_cosine": self._flat_cosine(real_tokens, repeat_tokens),
                "real_vs_zero_cosine": self._flat_cosine(real_tokens, zero_tokens),
                "repeat_current_vs_zero_cosine": self._flat_cosine(repeat_tokens, zero_tokens),
            },
        }

        stats_path = self.stats_dir / f"{frame_index:05}.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        if self.save_heatmaps:
            for name, values in weighted_delta_heatmaps.items():
                self._save_heatmap(values, self.heatmap_dir / f"{frame_index:05}_{name}.jpg")

    def _call_temporal_encoder(
        self,
        past_frame_features: torch.Tensor,
        current_frame_features: torch.Tensor,
    ) -> torch.Tensor:
        temporal_dtype = next(self.temporal_encoder.parameters()).dtype
        past_frame_features = past_frame_features.to(dtype=temporal_dtype)
        current_frame_features = current_frame_features.to(dtype=temporal_dtype)
        if getattr(self.temporal_encoder, "uses_current_frame", False):
            return self.temporal_encoder(
                past_frame_features,
                current_frame_tokens=current_frame_features,
            )
        return self.temporal_encoder(past_frame_features)

    def _gate_info(self):
        gate = getattr(self.temporal_encoder, "gate", None)
        gate_info = {
            "enabled": bool(getattr(self.temporal_encoder, "gate_enabled", False)),
        }
        if gate is not None:
            gate_float = gate.detach().float()
            gate_info["raw"] = float(gate_float.cpu())
            gate_info["sigmoid"] = float(torch.sigmoid(gate_float).cpu())
        return gate_info

    def _compute_weighted_delta_heatmaps(
        self,
        past_frame_features: torch.Tensor,
        current_frame_features: torch.Tensor,
    ):
        signed_delta = current_frame_features.unsqueeze(1).float() - past_frame_features.float()
        time_weights = self._time_weights(
            signed_delta.size(1),
            signed_delta.device,
            signed_delta.dtype,
        )
        weight_view = time_weights.view(1, -1, 1, 1)
        weight_sum = time_weights.sum().clamp_min(1e-12)

        weighted_signed_delta = (signed_delta * weight_view).sum(dim=1) / weight_sum
        weighted_abs_delta = (signed_delta.abs() * weight_view).sum(dim=1) / weight_sum
        heatmaps = {
            "weighted_signed_delta_norm": weighted_signed_delta.norm(dim=-1)[0],
            "weighted_abs_delta_norm": weighted_abs_delta.norm(dim=-1)[0],
            "latest_delta_norm": signed_delta[:, -1].norm(dim=-1)[0],
        }

        projected_motion_norm = self._projected_motion_norm(
            past_frame_features,
            current_frame_features,
        )
        if projected_motion_norm is not None:
            heatmaps["projected_motion_norm"] = projected_motion_norm

        return heatmaps

    def _time_weights(
        self,
        num_past_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if hasattr(self.temporal_encoder, "get_time_weights"):
            return self.temporal_encoder.get_time_weights(num_past_frames, device, dtype)
        return torch.ones(num_past_frames, device=device, dtype=dtype)

    def _projected_motion_norm(
        self,
        past_frame_features: torch.Tensor,
        current_frame_features: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        required_attrs = ("delta_projection", "motion_norm", "get_time_weights")
        if not all(hasattr(self.temporal_encoder, attr) for attr in required_attrs):
            return None

        temporal_dtype = next(self.temporal_encoder.parameters()).dtype
        was_training = self.temporal_encoder.training
        self.temporal_encoder.eval()
        try:
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                past_tokens = past_frame_features.to(dtype=temporal_dtype)
                current_tokens = current_frame_features.to(dtype=temporal_dtype).unsqueeze(1)
                delta_tokens = current_tokens - past_tokens
                if getattr(self.temporal_encoder, "include_absolute_delta", False):
                    delta_tokens = torch.cat((delta_tokens, delta_tokens.abs()), dim=-1)

                time_weights = self.temporal_encoder.get_time_weights(
                    delta_tokens.size(1),
                    device=delta_tokens.device,
                    dtype=delta_tokens.dtype,
                ).view(1, -1, 1, 1)
                shared_delta_features = (delta_tokens * time_weights).sum(dim=1) / time_weights.sum()
                motion_features = self.temporal_encoder.delta_projection(shared_delta_features)
                motion_features = self.temporal_encoder.motion_norm(motion_features)
            return motion_features.detach().float().norm(dim=-1)[0]
        finally:
            self.temporal_encoder.train(was_training)

    @staticmethod
    def _heatmap_stats(heatmaps):
        return {
            name: {
                "mean": float(values.detach().float().mean().cpu()),
                "max": float(values.detach().float().max().cpu()),
            }
            for name, values in heatmaps.items()
        }

    def _patch_values_to_grid(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        num_tokens = values.shape[0]
        side = math.isqrt(num_tokens)
        if side * side == num_tokens:
            return values.reshape(side, side)

        tokens_per_patch = int(self.num_image_tokens_per_patch)
        if tokens_per_patch > 0 and num_tokens % tokens_per_patch == 0:
            patch_side = math.isqrt(tokens_per_patch)
            if patch_side * patch_side == tokens_per_patch:
                num_patches = num_tokens // tokens_per_patch
                tiles = values.reshape(num_patches, patch_side, patch_side)
                return np.concatenate([tiles[idx] for idx in range(num_patches)], axis=1)

        return values.reshape(1, num_tokens)

    def _save_heatmap(self, values: torch.Tensor, out_path: Path) -> None:
        grid = self._patch_values_to_grid(values.detach().float().cpu().numpy())
        grid = grid - grid.min()
        denom = grid.max()
        if denom > 1e-12:
            grid = grid / denom
        heatmap = (grid * 255.0).clip(0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(
            heatmap,
            (max(1, heatmap.shape[1] * 16), max(1, heatmap.shape[0] * 16)),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imwrite(str(out_path), heatmap)

    @staticmethod
    def _mean_token_l2(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left.detach().float() - right.detach().float()).norm(dim=-1).mean().cpu())

    @staticmethod
    def _flat_cosine(left: torch.Tensor, right: torch.Tensor):
        left = left.detach().float().flatten()
        right = right.detach().float().flatten()
        denom = left.norm() * right.norm()
        if float(denom.cpu()) <= 1e-12:
            return None
        return float(((left * right).sum() / denom).cpu())

    @staticmethod
    def _token_norm_stats(tokens: torch.Tensor):
        norms = tokens.detach().float().norm(dim=-1)
        return {
            "mean": float(norms.mean().cpu()),
            "max": float(norms.max().cpu()),
        }
