import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from configs.config import AeroConfig
from .explorer import OracleCandidates, SyntheticCandidates


class OnlineDPOExplorer:
    """
    Vanilla online DPO baseline: no ensemble, no routing.
    Randomly selects 2 responses per prompt, sends all to oracle.
    """

    def __init__(self, config: AeroConfig, step: int = 0):
        self.config = config
        self.step = step

    def select_candidates(
        self,
        rewards: torch.Tensor,          # ignored
        prompts: List[str],
        responses: Dict[int, List[str]],
    ) -> Tuple[OracleCandidates, None]:
        M = len(prompts)
        if M == 0:
            return OracleCandidates([], {}, []), None

        N = len(responses[0])

        oracle_prompts = []
        oracle_candidates = {}
        oracle_meta = []

        for i in range(M):
            a, b = random.sample(range(N), 2)
            oracle_prompts.append(prompts[i])
            oracle_candidates[len(oracle_prompts) - 1] = [
                responses[i][a],
                responses[i][b],
            ]
            oracle_meta.append((i, a, b))

        oracle_data = OracleCandidates(
            prompts=oracle_prompts,
            candidates=oracle_candidates,
            meta=oracle_meta,
        )
        return oracle_data, None
