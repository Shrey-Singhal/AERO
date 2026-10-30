# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is adapted from Oat (https://github.com/sail-sg/oat),
# oat/oat/rm/model.py. Modified by the AERO authors in 2026.

from typing import Dict, List
import torch
from torch import nn, optim
import einops
from configs.config import AeroConfig
from .ensemble import EnsembleModel
from .RMbackbone import RMbackbone
from ..preference.preference_learning import PairWiseLoss, UniformBuffer, RewardBatch
from ..dataset_utils import PromptInput


class RewardModel(nn.Module):

    def __init__(self, config: AeroConfig):
        super().__init__()
        self.config = config
        self.device = getattr(config, 'device_erm', config.device_actor)

        self.backbone = RMbackbone(config)

        self.model = EnsembleModel(
            encoding_dim=self.config.encoding_dim,
            num_ensemble=self.config.num_ensemble,
            hidden_dim = self.config.rm_hidden_dim,
            activation=self.config.rm_act_fn,
        )
        self.model.init()
        #move ensemble to GPU
        self.model.to(self.device)

        self.backbone.to(self.device)
        # Verify backbone actually landed on the right device
        bb_device = next(self.backbone.parameters()).device
        if str(bb_device) != str(self.device):
            print(f"[RewardModel] WARNING: backbone on {bb_device}, expected {self.device}. Forcing move.")
            self.backbone = self.backbone.to(torch.device(self.device))
        print(f"[RewardModel] ERM backbone + ensemble loaded on {self.device}")

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.rm_lr, weight_decay=self.config.rm_wd
        )
        self.reg_lambda = self.config.enn_lambda
        self.sgd_steps = self.config.rm_sgd_steps
        

    @torch.no_grad()
    def get_rewards(self, prompts: List[PromptInput], responses: Dict[int, List[str]]) -> torch.Tensor:
        features = self.get_features(prompts, responses)
        M, N, D = features.shape
        E = self.config.num_ensemble
        features = einops.rearrange(features, "m n d -> (m n) d")
        rewards = []

        for ndx in range(0, len(features), self.config.backbone_batch):
            batch_feat = features[ndx : min(ndx + self.config.backbone_batch, len(features))]
            batch_feat = batch_feat.to(next(self.model.parameters()).dtype)
            batch_feat = batch_feat[None, :, :].repeat([E, 1, 1])
            rewards.append(self.model(batch_feat)) #per batch
        rewards = torch.cat(rewards, dim=1)  # (E, M*N, 1)
        rewards = rewards.view(E, M, N, 1)
        return rewards

    def _batch_tokenize_pairs(
        self,
        prompts: List[PromptInput],
        responses: List[str],
    ) -> Dict[str, torch.Tensor]:
        all_tokenized = [
            self.backbone.tokenize_pair(prompt, response)
            for prompt, response in zip(prompts, responses)
        ]
        return self.backbone.tokenizer.pad(
            {"input_ids": all_tokenized},
            return_tensors="pt",
        )

    def get_features(self, prompts: List[PromptInput], responses: Dict[int, List[str]]):
        M = len(prompts)
        N = len(responses[0])

        flat_prompts: List[str] = []
        flat_responses: List[str] = []
        for i in range(M):
            for response in responses[i]:
                flat_prompts.append(prompts[i])
                flat_responses.append(response)

        encs = self._batch_tokenize_pairs(flat_prompts, flat_responses)

        device = next(self.backbone.parameters()).device
        total_pairs = M * N
        features = []
        batch_size = max(1, int(self.config.backbone_batch))
        ndx = 0
        while ndx < total_pairs:
            batch_end_ndx = min(ndx + batch_size, total_pairs)

            batch_enc = {}
            for key, tensor in encs.items():
                batch_enc[key] = tensor[ndx:batch_end_ndx].to(device)

            with torch.no_grad():
                try:
                    batch_features = self.backbone(**batch_enc)
                except RuntimeError as e:
                    err_msg = str(e).lower()
                    if "out of memory" in err_msg and batch_end_ndx - ndx > 1:
                        new_batch_size = max(1, (batch_end_ndx - ndx) // 2)
                        print(
                            f"[RM] WARNING: OOM at backbone_batch={batch_end_ndx - ndx}, "
                            f"retrying with {new_batch_size}"
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        batch_size = new_batch_size
                        continue
                    if "Prefix tokens" not in str(e):
                        raise
                    # Repair: re-tokenize bad samples with explicit prefix IDs
                    print(f"[RM] WARNING: repairing malformed samples in batch {ndx}-{batch_end_ndx}")
                    repaired = []
                    for j in range(ndx, batch_end_ndx):
                        prompt_idx, resp_idx = j // N, j % N
                        ids = self.backbone.tokenize_pair_safe(
                            prompts[prompt_idx],
                            responses[prompt_idx][resp_idx],
                        )
                        repaired.append(ids)
                    rep_enc = self.backbone.tokenizer.pad(
                        {"input_ids": repaired},
                        return_tensors="pt",
                    )
                    rep_enc = {k: v.to(device) for k, v in rep_enc.items()}
                    batch_features = self.backbone(**rep_enc)
                features.append(batch_features)
            ndx = batch_end_ndx

        features = torch.cat(features, dim=0)  # (M*N, 2*D)
        
        # Reshape to (Prompts, Candidates, Features) -> (M, N, 2*D)
        features = features.view(M, N, -1)

        return features

    def learn_from_buffer(self, buffer: UniformBuffer, train_bs: int) -> dict:
        """Train ensemble RM on buffer data."""
        self.model.train()
        self.backbone.eval()

        loss_fn = PairWiseLoss()
        total_num_queries = max(1, int(buffer.total_num_queries))
        eff_bs = min(int(train_bs), int(buffer.size))

        for _ in range(self.sgd_steps):
            batch = buffer.sample(eff_bs)

            # Prepare inputs: (B, 2, D) -> (2B, D) -> (E, 2B, D)
            pair_feats = batch.pair_features.view(2 * eff_bs, -1)
            E = self.config.num_ensemble
            batch_inp = pair_feats[None, :, :].repeat(E, 1, 1)

            # Forward pass
            scores = self.model(batch_inp)                      # (E, 2B, 1)
            scores = scores.view(E, eff_bs, 2, 1)               # (E, B, 2, 1)
            
            chosen_scores = scores[:, :, 0, :]                  # (E, B, 1)
            rejected_scores = scores[:, :, 1, :]                # (E, B, 1)

            mask = batch.loss_masks[None, :]                    

            # Loss calculation
            loss_rew = loss_fn(chosen_scores, rejected_scores, mask)
            
            # Regularization: lambda * (B / N) * Prior
            loss_reg = self.reg_lambda * (eff_bs / total_num_queries) * self.model.regularization()

            self.optimizer.zero_grad()
            (loss_rew + loss_reg).backward()
            self.optimizer.step()

        return {
            "rm/loss_rew": float(loss_rew.detach().cpu()),
            "rm/loss_reg": float(loss_reg.detach().cpu()),
            "rm/chosen_rewards": float(chosen_scores.mean().detach().cpu()),
            "rm/rejected_rewards": float(rejected_scores.mean().detach().cpu()),
            "rm/lambda_scaled": float(self.reg_lambda * eff_bs / total_num_queries),
            "rm/total_queries": int(buffer.total_num_queries),
            "rm/buffer_size": int(buffer.size),
            "rm/updated": True,
        }
