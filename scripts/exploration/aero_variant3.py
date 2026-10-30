"""
AERO Variant 3 Explorer.

Same as Explorer (AERO) but synthetic response selection and labeling
follows SEA's approach:
- Single random ensemble head per prompt
- Max-margin pair selection from that head
- Label from the SAME single head (not ensemble mean)
- Aleatoric filtering is still applied
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

from configs.config import AeroConfig
from ..reward.reward_model import RewardModel


@dataclass
class OracleCandidates:
    """Data for samples to be labeled by oracle."""
    prompts: List[str]
    candidates: Dict[int, List[str]]  # idx -> [response_a, response_b]
    meta: List[Tuple[int, int, int]]  # (orig_prompt_idx, anchor_idx, challenger_idx)
    reference_responses: List[str] = field(default_factory=list)


@dataclass
class SyntheticCandidates:
    """Data for samples labeled by a sampled reward head."""
    prompts: List[str]
    candidates: Dict[int, List[str]]  # idx -> [response_a, response_b]
    probs: torch.Tensor  # sampled-head probabilities P(A > B)

class AeroVariant3Explorer:
    def __init__(self, rm: RewardModel, config: AeroConfig, step: int = 0):
        self.rm = rm
        self.config = config
        self.step = step
        self.device = config.device_actor
        self.burn_in_period = config.burn_in_period
        self._diagnostics = {}

    @property
    def is_burn_in(self) -> bool:
        return self.step < self.burn_in_period

    @staticmethod
    def _binary_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))

    def _otsu_threshold_with_quality(self, scores: torch.Tensor) -> Tuple[float, float]:
        if scores.numel() < 2:
            return float("inf"), 0.0
        vals, _ = torch.sort(scores)
        total_var = vals.var(unbiased=False).item()
        total = vals.sum()
        cumsum = torch.cumsum(vals, 0)
        n = vals.numel()
        idx = torch.arange(1, n, device=vals.device)
        w0 = idx / n
        w1 = 1.0 - w0
        mu0 = cumsum[:-1] / idx
        mu1 = (total - cumsum[:-1]) / (n - idx)
        between = w0 * w1 * (mu0 - mu1) ** 2
        best = torch.argmax(between).item()
        threshold = float((vals[best] + vals[best + 1]) / 2.0)
        quality = float(between[best].item()) / (total_var + 1e-8)
        return threshold, quality

    def _compute_vote_entropy_and_agreement(self, r: torch.Tensor) -> Tuple[torch.Tensor, float]:
        E, M, N = r.shape
        argmax_e = r.argmax(dim=2)  # (E, M)

        counts = torch.zeros((M, N), device=r.device)
        idx = argmax_e.transpose(0, 1)  # (M, E)
        counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=counts.dtype))

        probs = counts / float(E)  # (M, N)
        vote_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # (M,)

        agreement = probs.max(dim=1).values
        ens_argmax_agreement_mean = float(agreement.mean().item())

        return vote_entropy, ens_argmax_agreement_mean

    def _ts_anchor(self, r: torch.Tensor) -> torch.Tensor:
        E, M, N = r.shape
        sampled_heads = torch.randint(E, (M,), device=r.device)
        sampled_r = r[sampled_heads, torch.arange(M, device=r.device), :]
        return sampled_r.argmax(dim=1)

    def _route_prompts(
        self, scores: torch.Tensor, M: int
    ) -> Tuple[List[int], List[int], float, float, float]:
        threshold, quality = self._otsu_threshold_with_quality(scores)
        oracle_mask = scores > threshold
        otsu_ratio = float(oracle_mask.float().mean().item()) if M > 0 else 0.0
        oracle_indices = torch.where(oracle_mask)[0]
        synthetic_indices = torch.where(~oracle_mask)[0]

        return (
            oracle_indices.tolist(),
            synthetic_indices.tolist(),
            float(threshold),
            float(quality),
            otsu_ratio,
        )

    def select_candidates(
        self,
        rewards: torch.Tensor,                 # (E, M, N, 1)
        prompts: List[str],
        responses: Dict[int, List[str]]
    ):
        _, M, _, _ = rewards.shape
        if M == 0:
            return OracleCandidates([], {}, []), None

        r = rewards.squeeze(-1)  # (E, M, N)
        E, M, N = r.shape

        # Pairwise stats
        delta_all = r[:, :, :, None] - r[:, :, None, :]      # (E, M, N, N)
        p_e_all = torch.sigmoid(delta_all)                   # (E, M, N, N)
        p_bar_all = p_e_all.mean(dim=0)                      # (M, N, N)
        mean_H_all = self._binary_entropy(p_e_all).mean(dim=0)  # (M, N, N)
        var_all = delta_all.var(dim=0)                       # (M, N, N)

        # Routing score
        vote_entropy, ens_agree_mean = self._compute_vote_entropy_and_agreement(r)

        routing_scores = vote_entropy

        # Burn-in: all oracle
        if self.is_burn_in:
            oracle_indices = list(range(M))
            synthetic_indices = []
            otsu_threshold, otsu_quality = self._otsu_threshold_with_quality(routing_scores)
            otsu_ratio = float((routing_scores > otsu_threshold).float().mean().item())
        else:
            oracle_indices, synthetic_indices, otsu_threshold, otsu_quality, otsu_ratio = self._route_prompts(routing_scores, M)

        # Thompson-sampled anchor for oracle-routed prompts
        anchor_idx = self._ts_anchor(r)  # (M,)

        # Synthetic selection: SEA-style (single head, max-margin)
        # with aleatoric filtering
        syn_selected = []
        syn_pairs = []
        syn_dropped = 0

        ale_thr = self.config.aleatoric_threshold

        # Compute pairwise reward margins: (E, M, N, N)
        reward_margin = rewards.squeeze(-1)[:, :, :, None] - rewards.squeeze(-1)[:, :, None, :]

        # Sample one ensemble head per synthetic prompt (SEA-style)
        num_synthetic = len(synthetic_indices)
        if num_synthetic > 0:
            head_indices = torch.randint(E, (num_synthetic,), device=rewards.device)

        for new_idx, orig_idx in enumerate(synthetic_indices):
            h = head_indices[new_idx].item()

            # Get reward margin matrix for this prompt from sampled head
            margin_i = reward_margin[h, orig_idx, :, :]  # (N, N)
            margin_i_abs = torch.abs(margin_i)

            # Apply aleatoric filtering: mask pairs with high mean entropy
            if self.config.aleatoric_filter and not self.is_burn_in:
                aleatoric_mask = mean_H_all[orig_idx] > ale_thr  # (N, N)
                margin_i_abs[aleatoric_mask] = float("-inf")

            # Exclude self-pairs
            margin_i_abs.fill_diagonal_(float("-inf"))

            if not torch.isfinite(margin_i_abs).any():
                syn_dropped += 1
                continue

            # Find max-margin pair (SEA-style)
            max_val = margin_i_abs.max()
            tr_pairs = torch.where(margin_i_abs == max_val)

            # Break ties randomly
            sel_idx = np.random.choice(len(tr_pairs[0]))
            cand_a = tr_pairs[0][sel_idx].item()
            cand_b = tr_pairs[1][sel_idx].item()

            # Determine winner based on margin sign from the SAME sampled head
            if margin_i[cand_a, cand_b] > 0:
                winner, loser = cand_a, cand_b
            else:
                winner, loser = cand_b, cand_a

            # Label using the SAME single head (SEA-style, not ensemble mean)
            diff = r[h, orig_idx, winner] - r[h, orig_idx, loser]
            prob = float(torch.sigmoid(diff).item())

            syn_selected.append(orig_idx)
            syn_pairs.append((orig_idx, winner, loser, prob))

        # Select the max-variance challenger; exclude aleatoric pairs after burn-in
        oracle_used = []
        oracle_pairs = []
        oracle_dropped = 0

        for orig_idx in oracle_indices:
            a = int(anchor_idx[orig_idx].item())

            row_var = var_all[orig_idx, a, :].clone()
            row_H = mean_H_all[orig_idx, a, :]
            if self.is_burn_in or not self.config.aleatoric_filter:
                invalid = torch.zeros_like(row_var, dtype=torch.bool)
            else:
                invalid = (row_H > ale_thr)
            invalid[a] = True
            row_var[invalid] = float("-inf")

            if not torch.isfinite(row_var).any():
                oracle_dropped += 1
                continue

            b = int(row_var.argmax().item())
            oracle_used.append(orig_idx)
            oracle_pairs.append((orig_idx, a, b))

        # Pack outputs
        # OracleCandidates
        oracle_prompts = []
        oracle_candidates = {}
        oracle_meta = []

        for out_i, (orig_idx, a, b) in enumerate(oracle_pairs):
            oracle_prompts.append(prompts[orig_idx])
            oracle_candidates[out_i] = [responses[orig_idx][a], responses[orig_idx][b]]
            oracle_meta.append((orig_idx, a, b))

        oracle_data = OracleCandidates(
            prompts=oracle_prompts,
            candidates=oracle_candidates,
            meta=oracle_meta,
        )

        # SyntheticCandidates
        synthetic_data = None
        if len(syn_pairs) > 0:
            syn_prompts = []
            syn_candidates = {}
            syn_probs = []

            for out_i, (orig_idx, winner, loser, prob) in enumerate(syn_pairs):
                syn_prompts.append(prompts[orig_idx])
                syn_candidates[out_i] = [responses[orig_idx][winner], responses[orig_idx][loser]]
                syn_probs.append(torch.tensor(prob, device=r.device))

            synthetic_data = SyntheticCandidates(
                prompts=syn_prompts,
                candidates=syn_candidates,
                probs=torch.stack(syn_probs),
            )

        # Binary BALD for chosen pairs
        def _pair_binary_bald(orig_idx, a, b):
            p_bar_ab = p_bar_all[orig_idx, a, b]
            mean_H_ab = mean_H_all[orig_idx, a, b]
            H_p_bar = self._binary_entropy(p_bar_ab)
            return float((H_p_bar - mean_H_ab).item())

        oracle_bald_list = [_pair_binary_bald(orig_idx, a, b) for (orig_idx, a, b) in oracle_pairs]
        oracle_bald_mean = sum(oracle_bald_list) / len(oracle_bald_list) if oracle_bald_list else 0.0

        syn_bald_list = [_pair_binary_bald(orig_idx, w, l) for (orig_idx, w, l, _) in syn_pairs]
        syn_bald_mean = sum(syn_bald_list) / len(syn_bald_list) if syn_bald_list else 0.0

        # Diagnostics
        o_t = torch.tensor(oracle_used, device=r.device) if len(oracle_used) > 0 else None
        s_t = torch.tensor(syn_selected, device=r.device) if len(syn_selected) > 0 else None

        self._diagnostics = {
            "is_burn_in": self.is_burn_in,
            "num_prompts": M,

            "num_oracle": len(oracle_used),
            "num_synthetic": len(syn_selected),

            "binary_bald_mean_oracle": oracle_bald_mean,
            "binary_bald_mean_synth": syn_bald_mean,

            "dropped_total": int(oracle_dropped + syn_dropped),
            "dropped_oracle_no_safe_pair": int(oracle_dropped),
            "dropped_synth_no_safe_pair": int(syn_dropped),

            "oracle_ratio": round(len(oracle_used) / M, 4) if M > 0 else 0.0,

            "routing_metric": "vote_entropy",
            "vote_entropy_mean": float(vote_entropy.mean().item()),
            "vote_entropy_mean_oracle": float(vote_entropy[o_t].mean().item()) if o_t is not None else 0.0,
            "vote_entropy_mean_synth": float(vote_entropy[s_t].mean().item()) if s_t is not None else 0.0,

            "ens_argmax_agreement_mean": ens_agree_mean,

            "aleatoric_threshold": ale_thr,

            "otsu_threshold": float(otsu_threshold),
            "otsu_quality": float(otsu_quality),
            "otsu_ratio": float(otsu_ratio),
        }

        return oracle_data, synthetic_data

    def get_diagnostics(self) -> dict:
        return self._diagnostics.copy()
