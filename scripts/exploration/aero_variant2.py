"""AERO Variant 2: fixed-ratio random routing after burn-in."""

from __future__ import annotations

import torch

from .explorer import Explorer


class AeroVariant2Explorer(Explorer):
    """Randomly assign a fixed fraction of post-burn-in prompts to the oracle branch."""

    def _target_oracle_count(self, num_prompts: int) -> int:
        ratio = float(self.config.aero_variant2_oracle_ratio)
        return max(0, min(num_prompts, int(round(ratio * num_prompts))))

    def _route_prompts(self, scores: torch.Tensor, M: int):
        if M <= 0:
            return [], [], 0.0, 0.0, 0.0

        target_oracle = self._target_oracle_count(M)
        perm = torch.randperm(M, device=scores.device)
        oracle_indices = perm[:target_oracle]
        synthetic_indices = perm[target_oracle:]
        actual_ratio = float(target_oracle / M)

        return (
            oracle_indices.tolist(),
            synthetic_indices.tolist(),
            0.0,
            0.0,
            actual_ratio,
        )

    def select_candidates(self, rewards, prompts, responses):
        oracle_data, synthetic_data = super().select_candidates(
            rewards, prompts, responses
        )
        self._diagnostics.pop("otsu_threshold", None)
        self._diagnostics.pop("otsu_quality", None)
        self._diagnostics.pop("otsu_ratio", None)
        self._diagnostics["routing_metric"] = "fixed_random"
        self._diagnostics["fixed_oracle_ratio_target"] = float(
            self.config.aero_variant2_oracle_ratio
        )
        self._diagnostics["fixed_oracle_count_target"] = (
            self._target_oracle_count(len(prompts))
        )
        return oracle_data, synthetic_data
