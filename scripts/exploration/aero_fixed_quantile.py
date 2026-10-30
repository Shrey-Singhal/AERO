"""AERO with fixed-quantile vote-entropy routing."""

from __future__ import annotations

import torch

from .explorer import Explorer


class AeroFixedQuantileExplorer(Explorer):
    """Route the highest vote-entropy fraction to the oracle."""

    def _target_oracle_count(self, num_prompts: int) -> int:
        ratio = float(self.config.aero_fixed_quantile_ratio)
        return max(0, min(num_prompts, int(round(ratio * num_prompts))))

    def _route_prompts(self, scores: torch.Tensor, M: int):
        if M <= 0:
            return [], [], float("inf"), 0.0, 0.0

        target_oracle = self._target_oracle_count(M)

        # Randomize equal-entropy ties without advancing AERO's global RNG.
        tie_generator = torch.Generator(device=scores.device)
        tie_generator.manual_seed(
            int(self.config.seed) * 1_000_003 + int(self.step)
        )
        permutation = torch.randperm(
            M, device=scores.device, generator=tie_generator
        )
        ranked = permutation[
            torch.argsort(scores[permutation], descending=True, stable=True)
        ]
        oracle_indices = ranked[:target_oracle]
        synthetic_indices = ranked[target_oracle:]

        if target_oracle == 0:
            boundary_score = float("inf")
        elif target_oracle == M:
            boundary_score = float("-inf")
        else:
            boundary_score = float(scores[oracle_indices[-1]].item())

        actual_ratio = float(target_oracle / M)
        return (
            oracle_indices.tolist(),
            synthetic_indices.tolist(),
            boundary_score,
            0.0,
            actual_ratio,
        )

    def select_candidates(self, rewards, prompts, responses):
        oracle_data, synthetic_data = super().select_candidates(
            rewards, prompts, responses
        )
        boundary_score = self._diagnostics.pop("otsu_threshold", None)
        self._diagnostics.pop("otsu_quality", None)
        actual_ratio = self._diagnostics.pop("otsu_ratio", 0.0)
        if self.is_burn_in:
            boundary_score = float("-inf")
            actual_ratio = 1.0
        self._diagnostics.update(
            {
                "routing_metric": "fixed_quantile_vote_entropy",
                "fixed_quantile_target_ratio": float(
                    self.config.aero_fixed_quantile_ratio
                ),
                "fixed_quantile_actual_ratio": float(actual_ratio),
                "fixed_quantile_oracle_count_target": self._target_oracle_count(
                    len(prompts)
                ),
                "fixed_quantile_boundary_score": boundary_score,
            }
        )
        return oracle_data, synthetic_data
