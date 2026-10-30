import os
import random
from contextlib import nullcontext
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config import AeroConfig
from ..actor import Actor
from ..preference.preference_dataset import zero_pad_sequences
from ..reward.reward_model import RewardModel
from .explorer import OracleCandidates


class CountValueHead(nn.Module):
    """Count-value head adapted from the reference CoPO CountTrainer."""

    def __init__(self, hidden_size: int, summary_dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        self.hidden = nn.Linear(hidden_size, 32)
        self.summary = nn.Linear(32, 20)
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten()

    def forward(self, hidden_states, return_last: bool = True):
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        output = self.hidden(output)
        output = self.lrelu(output)
        output = self.summary(output)
        if return_last:
            output = output[..., -1, :]
        return output


class CoinFlipMaker:
    """Coin-flip target generator adapted from the reference CoPO CountTrainer."""

    def __init__(self, output_dimensions: int = 20):
        self.output_dimensions = output_dimensions

    def __call__(self, batch_size: int):
        return 2 * np.random.binomial(1, 0.5, size=(batch_size, self.output_dimensions)) - 1


class CoPOCFNTrainer:
    def __init__(self, config: AeroConfig, tokenizer, dpo_trainer, clean_memory=None):
        self.config = config
        self.tokenizer = tokenizer
        self.dpo_trainer = dpo_trainer
        self.clean_memory = clean_memory

    def fit_buffer(self, flat_examples: List[Dict[str, str]], total_iterations: int) -> None:
        if not self.config.use_copo or self.config.copo_mode != "copo_rolling":
            return
        if not flat_examples:
            return
        self._fit_examples(
            flat_examples,
            total_iterations=total_iterations,
            log_prefix="[CoPO CFN]",
            load_existing=True,
        )

    def _fit_examples(
        self,
        flat_examples: List[Dict[str, str]],
        total_iterations: int,
        log_prefix: str,
        load_existing: bool,
    ) -> None:
        print(f"{log_prefix} Training CFN on {len(flat_examples)} prompt/response pairs")

        self.dpo_trainer.ensure_initialized(total_iterations)
        policy_model = self.dpo_trainer.get_model()
        policy_was_training = policy_model.training
        policy_model.eval()

        model_device = next(policy_model.parameters()).device
        use_cuda = model_device.type == "cuda"
        autocast_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else None

        cfn_path = self._cfn_path()
        if load_existing and os.path.exists(cfn_path):
            try:
                cfn = torch.load(cfn_path, map_location=model_device, weights_only=False)
            except TypeError:
                cfn = torch.load(cfn_path, map_location=model_device)
        else:
            cfn = CountValueHead(hidden_size=int(policy_model.config.hidden_size)).to(model_device)

        print(
            f"{log_prefix} Hidden size check: model={policy_model.config.hidden_size}, "
            f"cfn={cfn.hidden.in_features}"
        )
        assert int(policy_model.config.hidden_size) == int(cfn.hidden.in_features)

        cfn.train()
        optimizer = torch.optim.Adam(cfn.parameters(), lr=float(self.config.copo_cfn_lr))
        coin_flip_maker = CoinFlipMaker()
        dataloader = DataLoader(
            flat_examples,
            batch_size=int(self.config.copo_cfn_batch_size),
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_batch,
        )

        losses: List[float] = []
        last_hidden_states = None

        for batch in dataloader:
            input_ids = batch["input_ids"].to(model_device)
            attention_mask = batch["attention_mask"].to(model_device)
            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=autocast_dtype)
                if autocast_dtype is not None
                else nullcontext()
            )

            with torch.no_grad():
                with autocast_ctx:
                    hidden_states = policy_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    ).hidden_states[-1]

            value_output = cfn(hidden_states)
            last_hidden_states = hidden_states.detach()
            targets = torch.from_numpy(coin_flip_maker(input_ids.shape[0])).to(
                device=model_device,
                dtype=value_output.dtype,
            )
            loss = F.mse_loss(value_output, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        cfn.eval()
        with torch.no_grad():
            sample_var = float(cfn(last_hidden_states).float().var().item())

        torch.save(cfn, cfn_path)
        print(
            f"{log_prefix} Completed: "
            f"mse_start={losses[0]:.6f}, mse_end={losses[-1]:.6f}, output_var={sample_var:.6f}"
        )

        del cfn
        if policy_was_training:
            policy_model.train()
        if hasattr(self.dpo_trainer, "release_local_model"):
            self.dpo_trainer.release_local_model()
        self._clean_memory()

    def collate_batch(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_masks = []

        for item in batch:
            response_text = (item["prompt"] + item["response"]).rstrip("\n")
            tok = self.tokenizer(
                response_text,
                max_length=self.config.prompt_max_length + self.config.generate_max_length - 1,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            eos_id = self.tokenizer.eos_token_id
            tok["input_ids"] = torch.cat([
                tok["input_ids"],
                torch.tensor([[eos_id]])
            ], dim=1)
            tok["attention_mask"] = torch.cat([
                tok["attention_mask"],
                torch.tensor([[1]])
            ], dim=1)
            input_ids.append(tok["input_ids"].squeeze(0))
            attention_masks.append(tok["attention_mask"].squeeze(0))

        pad_id = int(self.tokenizer.pad_token_id)
        return {
            "input_ids": zero_pad_sequences(input_ids, side="right", value=pad_id),
            "attention_mask": zero_pad_sequences(
                [mask.to(torch.long) for mask in attention_masks],
                side="right",
                value=0,
            ).bool(),
        }

    def _cfn_path(self) -> str:
        os.makedirs(self.config.output_dir, exist_ok=True)
        return os.path.join(self.config.output_dir, "vhead.pickle")

    def _clean_memory(self) -> None:
        if self.clean_memory is not None:
            self.clean_memory(light=False)


class CoPOStage1Trainer(CoPOCFNTrainer):
    def run(self, train_prompts: List[str], total_iterations: int) -> None:
        if not self.config.use_copo or self.config.copo_mode != "copo_fixed":
            return

        num_prompts = min(int(self.config.copo_stage1_num_prompts), len(train_prompts))
        stage1_prompts = train_prompts[:num_prompts]
        if not stage1_prompts:
            return

        print(f"\n[CoPO Stage 1] Generating {num_prompts} SFT reference responses")

        actor = None
        try:
            actor = Actor(
                self.config,
                eval_mode=False,
                num_samples_override=1,
                model_path_override=self.config.pretrain_model,
            )
            stage1_responses = actor.generate(stage1_prompts)
        finally:
            if actor is not None:
                actor.cleanup()
                del actor
            self._clean_memory()

        flat_examples = []
        for idx, prompt in enumerate(stage1_prompts):
            response_list = stage1_responses.get(idx, [])
            if not response_list:
                continue
            flat_examples.append({"prompt": prompt, "response": response_list[0]})

        if not flat_examples:
            raise RuntimeError("CoPO Stage 1 generated no reference responses.")

        self._fit_examples(
            flat_examples,
            total_iterations=total_iterations,
            log_prefix="[CoPO Stage 1]",
            load_existing=False,
        )


class CoPOExplorer:
    """
    CoPO explorer adapted for on-policy rollouts.

    Modes:
    - `copo_fixed`: fixed CFN with one-shot stage-1 pretraining.
    - `copo_rolling`: online burn-in + rolling CFN buffer updates.
    """

    def __init__(self, rm: RewardModel, config: AeroConfig, step: int = 0):
        self.rm = rm
        self.config = config
        self.step = step
        self._diagnostics = {}

    @property
    def is_rolling_mode(self) -> bool:
        return self.config.copo_mode == "copo_rolling"

    @property
    def is_burn_in(self) -> bool:
        return self.is_rolling_mode and self.step < int(self.config.burn_in_period)

    def select_candidates(
        self,
        rewards: torch.Tensor,
        prompts: List[str],
        responses: Dict[int, List[str]],
    ) -> Tuple[OracleCandidates, None]:
        num_prompts = len(prompts)
        if num_prompts == 0:
            return OracleCandidates([], {}, []), None

        if self.is_burn_in:
            oracle_prompts: List[str] = []
            oracle_candidates: Dict[int, List[str]] = {}
            oracle_meta: List[Tuple[int, int, int]] = []

            for prompt_idx in range(num_prompts):
                prompt_responses = responses[prompt_idx]
                assert len(prompt_responses) >= 2, "CoPO burn-in requires at least 2 responses per prompt."
                out_idx = len(oracle_prompts)
                oracle_prompts.append(prompts[prompt_idx])
                oracle_candidates[out_idx] = [prompt_responses[0], prompt_responses[1]]
                oracle_meta.append((prompt_idx, 0, 1))

            self._diagnostics = {
                "is_burn_in": True,
                "num_prompts": num_prompts,
                "num_oracle": num_prompts,
                "num_synthetic": 0,
                "max_margin_spread": 0.0,
            }
            return OracleCandidates(oracle_prompts, oracle_candidates, oracle_meta), None

        assert rewards is not None, "Post-burn-in CoPO requires reward model scores."
        _, _, num_candidates, _ = rewards.shape
        assert num_candidates >= 3, "CoPO requires at least 3 candidates per prompt after burn-in."
        mean_rewards = rewards.mean(dim=0).squeeze(-1)

        oracle_prompts: List[str] = []
        oracle_candidates: Dict[int, List[str]] = {}
        oracle_meta: List[Tuple[int, int, int]] = []
        reference_responses: List[str] = []

        for prompt_idx in range(num_prompts):
            ranking = torch.argsort(mean_rewards[prompt_idx])
            rejected_idx = int(ranking[0].item())
            chosen_idx = int(ranking[-1].item())
            remaining = [idx for idx in range(num_candidates) if idx not in (chosen_idx, rejected_idx)]

            assert chosen_idx != rejected_idx, "CoPO chosen/rejected indices must be distinct."
            assert remaining, "CoPO requires a distinct reference response."

            reference_idx = random.choice(remaining)
            assert reference_idx not in (chosen_idx, rejected_idx)

            out_idx = len(oracle_prompts)
            oracle_prompts.append(prompts[prompt_idx])
            oracle_candidates[out_idx] = [
                responses[prompt_idx][chosen_idx],
                responses[prompt_idx][rejected_idx],
            ]
            oracle_meta.append((prompt_idx, chosen_idx, rejected_idx))
            reference_responses.append(responses[prompt_idx][reference_idx])

        self._diagnostics = {
            "is_burn_in": False,
            "num_prompts": num_prompts,
            "num_oracle": num_prompts,
            "num_synthetic": 0,
            "max_margin_spread": float(
                (mean_rewards.max(dim=1).values - mean_rewards.min(dim=1).values).mean().item()
            ),
        }

        return (
            OracleCandidates(
                prompts=oracle_prompts,
                candidates=oracle_candidates,
                meta=oracle_meta,
                reference_responses=reference_responses,
            ),
            None,
        )

    def get_diagnostics(self) -> dict:
        return self._diagnostics.copy()

    def build_cfn_training_examples(
        self,
        prompts: List[str],
        responses: Dict[int, List[str]],
        oracle_candidates: OracleCandidates,
    ) -> List[Dict[str, str]]:
        if not self.is_rolling_mode:
            return []

        if self.is_burn_in:
            examples: List[Dict[str, str]] = []
            for prompt_idx, prompt in enumerate(prompts):
                if not responses[prompt_idx]:
                    continue
                examples.append({"prompt": prompt, "response": responses[prompt_idx][0]})
            return examples

        examples = []
        for out_idx, prompt in enumerate(oracle_candidates.prompts):
            if out_idx >= len(oracle_candidates.reference_responses):
                break
            examples.append(
                {
                    "prompt": prompt,
                    "response": oracle_candidates.reference_responses[out_idx],
                }
            )
        return examples
