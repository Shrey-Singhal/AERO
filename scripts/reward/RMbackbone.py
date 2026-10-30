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
# oat/oat/rm/backbone.py. Modified by the AERO authors in 2026.

import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model,
    DebertaV2PreTrainedModel,
)

from configs.config import AeroConfig
from scripts.hf_auth import hf_token_kwargs
from scripts.dataset_utils import (
    PromptInput,
    flatten_prompt_input,
)


class DebertaV2PairRM(DebertaV2PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.pretrained_model = DebertaV2Model(config)
        self.post_init()

    def forward(self, *args, **kwargs):
        return self.pretrained_model(*args, **kwargs)


class RMbackbone(nn.Module):
    def __init__(self, config: AeroConfig):
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.encoder,
            trust_remote_code=True,
            **hf_token_kwargs(),
        )

        # Load using wrapper, and VERIFY load success
        self.model, info = DebertaV2PairRM.from_pretrained(
            self.config.encoder,
            output_loading_info=True,
            **hf_token_kwargs(),
        )

        missing = info.get("missing_keys", [])
        unexpected = info.get("unexpected_keys", [])

        print(f"[PairRM load] missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0 or len(unexpected) > 0:
            print("  missing sample:", list(missing)[:10])
            print("  unexpected sample:", list(unexpected)[:10])

        # Take IDs from config
        self.source_prefix_id = self.model.config.source_prefix_id
        self.cand_prefix_id = self.model.config.cand_prefix_id
        # SEA uses literal prefix strings in tokenizer.encode.
        self.source_prefix = "<|source|>"
        self.cand_prefix = "<|candidate|>"
        print(f"  source_prefix_id={self.source_prefix_id}, cand_prefix_id={self.cand_prefix_id}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def tokenize_pair(self, prompt: PromptInput, response: str):
        # Match SEA: tokenizer encodes prefix strings directly and uses default special tokens.
        prompt_text = flatten_prompt_input(prompt)
        prompt_ids = self.tokenizer.encode(
            self.source_prefix + prompt_text,
            max_length=self.config.source_max_length,
            truncation=True,
        )

        cand_max = max(self.config.max_length - len(prompt_ids), 0)
        resp_ids = self.tokenizer.encode(
            self.cand_prefix + response,
            max_length=cand_max,
            truncation=True,
        )

        return prompt_ids + resp_ids

    def tokenize_pair_safe(self, prompt: PromptInput, response: str):
        """Fallback tokenizer that guarantees prefix tokens via explicit ID insertion."""
        prompt_text = flatten_prompt_input(prompt)
        prompt_ids = self.tokenizer.encode(
            prompt_text, add_special_tokens=False,
            max_length=self.config.source_max_length - 1, truncation=True,
        )
        remaining = max(self.config.max_length - len(prompt_ids) - 2, 1)
        resp_ids = self.tokenizer.encode(
            response, add_special_tokens=False,
            max_length=remaining, truncation=True,
        )
        return [self.source_prefix_id] + prompt_ids + [self.cand_prefix_id] + resp_ids

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Match SEA: drop all-pad columns before encoding.
        keep_column_mask = attention_mask.ne(0).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        encs = outputs.hidden_states[-1]

        source_idxs = torch.where(input_ids == self.source_prefix_id)
        cand_idxs = torch.where(input_ids == self.cand_prefix_id)

        if len(source_idxs[0]) != input_ids.shape[0] or len(cand_idxs[0]) != input_ids.shape[0]:
            raise RuntimeError("Prefix tokens not found exactly once per sample.")

        source_encs = encs[source_idxs[0], source_idxs[1], :]
        cand_encs = encs[cand_idxs[0], cand_idxs[1], :]

        return torch.cat([source_encs, cand_encs], dim=-1)
