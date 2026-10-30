"""
Preference Learning Buffers and Loss Functions
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RewardBatch:
    """Batch for ensemble reward model training."""
    pair_features: torch.Tensor  # shape - (B, 2, d). Represents feature vectors for chosen and rejected 
    loss_masks: torch.Tensor     # tensor of shape (B,) which is used to ignore certain samples

class UniformBuffer:
    """Tensor buffer for ensemble reward model. The buffer represents memory of past labels."""

    def __init__(self, max_len: int):
        self._max_len = int(max_len)
        self._pair = None
        self._mask = None
        self._n = 0     #current size
        self._idx = 0   #write pointer
        self.total_num_queries = 1

    def extend(self, batch: RewardBatch):
        if self._pair is None:
            d = batch.pair_features.shape[-1]
            device = batch.pair_features.device
            dtype = batch.pair_features.dtype
            self._pair = torch.empty((self._max_len, 2, d), device=device, dtype=dtype)
            self._mask = torch.empty((self._max_len,), device=device, dtype=batch.loss_masks.dtype)

        B = batch.pair_features.shape[0]
        idx = (torch.arange(self._idx, self._idx + B, device=batch.pair_features.device) % self._max_len)
        self._pair[idx] = batch.pair_features
        self._mask[idx] = batch.loss_masks
        self._idx = (self._idx + B) % self._max_len
        self._n = min(self._n + B, self._max_len)

    def sample(self, batch_size: int) -> Optional[RewardBatch]:
        if batch_size > self._n:
            return None
        start = np.random.choice(self._n, batch_size, replace=False)
        base = 0 if self._n < self._max_len else self._idx
        idx = (torch.tensor(start, device=self._pair.device) + base) % self._max_len
        return RewardBatch(self._pair[idx], self._mask[idx])

    @property
    def size(self) -> int:
        return self._n


class PairWiseLoss(torch.nn.Module):
    """Bradley-Terry pairwise preference loss."""

    def forward(self, chosen_reward, rejected_reward, mask):
        loss = -F.logsigmoid(chosen_reward - rejected_reward)
        return (loss * mask[..., None]).mean()
