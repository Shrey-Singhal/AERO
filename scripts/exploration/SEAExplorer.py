"""
SEA (Sample-Efficient Alignment) Explorer Implementation.

Implements the BAI (Best Arm Identification) objective from the SEA paper:
- Thompson Sampling for anchor (best) response selection
- Maximum variance for challenger response selection

For synthetic/model-labeled samples:
- Max-margin pair selection using one sampled reward head
- Synthetic labeling using the same sampled head

Reference: "Sample-Efficient Alignment for LLMs" (Appendix D)
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from configs.config import AeroConfig
from ..reward.reward_model import RewardModel


@dataclass
class OracleCandidates:
    """Data for samples to be labeled by oracle."""
    prompts: List[str]
    candidates: Dict[int, List[str]]  # idx -> [response_a, response_b]
    meta: List[Tuple[int, int, int]]  # (orig_prompt_idx, anchor_idx, challenger_idx)


@dataclass
class SyntheticCandidates:
    """Data for samples labeled by a sampled reward head."""
    prompts: List[str]
    candidates: Dict[int, List[str]]  # idx -> [response_a, response_b]
    probs: torch.Tensor  # sampled-head probabilities P(A > B)


class SEAExplorer:

    def __init__(self, rm: RewardModel, config: AeroConfig, step: int = 0):
        self.rm = rm
        self.config = config
        self.step = step
        self.device = config.device_actor
        self.burn_in_period = config.burn_in_period
        self.max_model_data_ratio = config.max_model_data_ratio

    def select_candidates(
        self,
        rewards: torch.Tensor,  # (E, M, N, 1)
        prompts: List[str],
        responses: Dict[int, List[str]]
    ) -> Tuple[OracleCandidates, Optional[SyntheticCandidates]]:
        """
        Select candidates for both oracle and synthetic labeling.

        During burn-in: all samples go to oracle (no synthetic)
        After burn-in: (1 - max_model_data_ratio) to oracle, rest to synthetic

        Oracle samples use BAI selection (Thompson sampling + max variance).
        Synthetic samples use one sampled head for max-margin selection and labeling.

        Args:
            rewards: Ensemble rewards of shape (E, M, N, 1)
            prompts: List of M prompts
            responses: Dict mapping prompt index to list of N responses

        Returns:
            oracle_candidates: Samples to be labeled by oracle
            synthetic_candidates: Samples labeled by sampled reward heads (None during burn-in)
        """
        E, M, N, _ = rewards.shape

        if M == 0:
            return (
                OracleCandidates(prompts=[], candidates={}, meta=[]),
                None
            )

        # Determine oracle vs synthetic split
        if self.step < self.burn_in_period:
            num_synthetic = 0
        else:
            num_synthetic = int(M * self.max_model_data_ratio)

        # Create assignment flags and shuffle
        is_synthetic = [False] * M
        for i in range(num_synthetic):
            is_synthetic[i] = True
        random.shuffle(is_synthetic)

        oracle_indices = [i for i in range(M) if not is_synthetic[i]]
        synthetic_indices = [i for i in range(M) if is_synthetic[i]]

        # Select oracle candidates (BAI selection)
        oracle_data = self._select_oracle_candidates(
            rewards, prompts, responses, oracle_indices
        )

        # Select and label synthetic candidates using one sampled head per prompt
        synthetic_data = None
        if synthetic_indices:
            synthetic_data = self._select_synthetic_candidates(
                rewards, prompts, responses, synthetic_indices
            )

        return oracle_data, synthetic_data

    def _select_oracle_candidates(
        self,
        rewards: torch.Tensor,
        prompts: List[str],
        responses: Dict[int, List[str]],
        indices: List[int]
    ) -> OracleCandidates:
        """
        Select candidates for oracle labeling using BAI selection.

        BAI (Best Arm Identification):
        - Anchor: Thompson Sampling (sample random head, pick its best response)
        - Challenger: Maximum variance (most uncertain pair)
        """
        selected_prompts = []
        selected_candidates = {}
        selected_meta = []

        for new_idx, orig_idx in enumerate(indices):
            prompt = prompts[orig_idx]
            prompt_rewards = rewards[:, orig_idx, :, 0]  # (E, N)

            anchor_idx, challenger_idx = self._bai_response_selection(prompt_rewards)

            selected_prompts.append(prompt)
            selected_candidates[new_idx] = [
                responses[orig_idx][anchor_idx],
                responses[orig_idx][challenger_idx]
            ]
            selected_meta.append((orig_idx, anchor_idx, challenger_idx))

        return OracleCandidates(
            prompts=selected_prompts,
            candidates=selected_candidates,
            meta=selected_meta
        )

    def _select_synthetic_candidates(
        self,
        rewards: torch.Tensor,
        prompts: List[str],
        responses: Dict[int, List[str]],
        indices: List[int]
    ) -> SyntheticCandidates:
        """
        Select candidates for synthetic labeling using max-margin selection.

        Follows OAT's _random_select approach:
        1. Compute pairwise reward margins for all prompts
        2. Sample one ensemble head per prompt (vectorized)
        3. Find max-margin pair for each prompt
        4. Label using the SAME sampled head
        """
        E, M_total, N, _ = rewards.shape
        num_synthetic = len(indices)

        # Compute pairwise reward margins: (E, M, N, N)
        # reward_margin[e, m, i, j] = reward[e, m, i] - reward[e, m, j]
        reward_margin = rewards - rewards.permute(0, 1, 3, 2)

        # Sample one ensemble head per synthetic prompt (like OAT)
        head_indices = torch.randint(E, (num_synthetic,), device=rewards.device)

        selected_prompts = []
        selected_candidates = {}
        probs_list = []

        for new_idx, orig_idx in enumerate(indices):
            prompt = prompts[orig_idx]
            h = head_indices[new_idx].item()

            # Get reward margin matrix for this prompt from sampled head
            margin_i = reward_margin[h, orig_idx, :, :]  # (N, N)
            margin_i_abs = torch.abs(margin_i)

            # Find max-margin pair(s)
            max_val = margin_i_abs.max()
            tr_pairs = torch.where(margin_i_abs == max_val)

            # Break ties randomly (OAT style)
            sel_idx = np.random.choice(len(tr_pairs[0]))
            cand_a = tr_pairs[0][sel_idx].item()
            cand_b = tr_pairs[1][sel_idx].item()

            # Determine winner based on margin sign from the SAME sampled head
            if margin_i[cand_a, cand_b] > 0:
                anchor_idx, challenger_idx = cand_a, cand_b
            else:
                anchor_idx, challenger_idx = cand_b, cand_a

            diff = rewards[h, orig_idx, anchor_idx, 0] - rewards[h, orig_idx, challenger_idx, 0]
            prob = torch.sigmoid(diff)

            selected_prompts.append(prompt)
            selected_candidates[new_idx] = [
                responses[orig_idx][anchor_idx],
                responses[orig_idx][challenger_idx]
            ]
            probs_list.append(prob)

        probs = torch.stack(probs_list) if probs_list else torch.tensor([], device=self.device)

        return SyntheticCandidates(
            prompts=selected_prompts,
            candidates=selected_candidates,
            probs=probs
        )

    def _bai_response_selection(
        self,
        prompt_rewards: torch.Tensor  # (E, N)
    ) -> Tuple[int, int]:
        """
        BAI (Best Arm Identification) response selection.

        Follows OAT's EnnBAITS approach:
        1. Thompson Sampling for anchor: sample random head, pick its best response
        2. Maximum LOGITS variance for challenger (not sigmoid variance)

        Args:
            prompt_rewards: Rewards from all ensemble heads, shape (E, N)

        Returns:
            (anchor_idx, challenger_idx)
        """
        E, N = prompt_rewards.shape

        # Step 1: Thompson Sampling for anchor
        # Sample random ensemble head and pick its best response
        sampled_head = random.randint(0, E - 1)
        anchor_idx = prompt_rewards[sampled_head].argmax().item()

        # Step 2: Maximum LOGITS variance for challenger
        # We want variance over ensemble dimension
        anchor_rewards = prompt_rewards[:, anchor_idx]  # (E,)

        best_challenger_idx = None
        best_variance = -float('inf')

        for j in range(N):
            if j == anchor_idx:
                continue

            challenger_rewards = prompt_rewards[:, j]  # (E,)

            # Compute LOGITS variance (not sigmoid variance!)
            pref_logits = anchor_rewards - challenger_rewards  # (E,)
            variance = pref_logits.var().item()

            if variance > best_variance:
                best_variance = variance
                best_challenger_idx = j

        # Handle the single-response edge case
        if best_challenger_idx is None:
            other_indices = [j for j in range(N) if j != anchor_idx]
            if other_indices:
                best_challenger_idx = random.choice(other_indices)
            else:
                best_challenger_idx = anchor_idx

        return anchor_idx, best_challenger_idx
