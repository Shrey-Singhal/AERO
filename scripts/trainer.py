from __future__ import annotations

from typing import Dict, List, Any, Tuple, Union
import os
import json
import gc
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from configs.config import AeroConfig
from .actor import Actor
from .exploration.SEAExplorer import SEAExplorer
from .exploration.explorer import Explorer
from .exploration.online_dpo import OnlineDPOExplorer
from .exploration.copo_explorer import CoPOExplorer, CoPOCFNTrainer, CoPOStage1Trainer
from .exploration.aero_variant2 import AeroVariant2Explorer
from .exploration.aero_variant3 import AeroVariant3Explorer
from .exploration.aero_fixed_quantile import AeroFixedQuantileExplorer
from .preference.preference_oracle import Oracle
from .reward.reward_model import RewardModel
from .preference.preference_dataset import TextPreferenceDataset
from .preference.preference_optimisation import PreferenceOptimisation
from .preference.preference_optimisation import PersistentDPOTrainer
from .preference.preference_learning import UniformBuffer
from .evaluation import Evaluate
from .dataset_utils import build_prompt_records


class AEROTrainer:
    def __init__(self, config: AeroConfig):
        self.config = config
        self._set_seed(self.config.seed)
        self.device = self.config.device_actor

        # Policy tokenizer (decoder-only)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.pretrain_model,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.global_buffer = UniformBuffer(max_len=self.config.r_buffer_maxlen)

        self.dpo_trainer = PersistentDPOTrainer(self.config, self.tokenizer)
        print("[Init] Policy trainer: Legacy persistent DPO")
        self.copo_stage1_trainer = None
        self.copo_cfn_trainer = None
        self.copo_cfn_buffer = None
        if self.config.use_copo:
            if self.config.copo_mode == "copo_fixed":
                self.copo_stage1_trainer = CoPOStage1Trainer(
                    self.config,
                    self.tokenizer,
                    self.dpo_trainer,
                    clean_memory=self._clean_memory,
                )
            else:
                self.copo_cfn_trainer = CoPOCFNTrainer(
                    self.config,
                    self.tokenizer,
                    self.dpo_trainer,
                    clean_memory=self._clean_memory,
                )
                self.copo_cfn_buffer = deque(maxlen=int(self.config.copo_cfn_buffer_maxlen))

        self.oracle = Oracle(self.config)

        if self.config.use_online_dpo:
            self.rm = None
            self.ensemble_trainer = None
            print("[Init] Online DPO mode: skipping RM backbone + ensemble (not needed)")
        else:
            self.rm = RewardModel(self.config)
            self.ensemble_trainer = PreferenceOptimisation(self.config, self.tokenizer)

        # Load dataset
        print("\n[Loading Dataset]")
        load_kwargs = {}
        if self.config.prompt_data_dir:
            load_kwargs["data_dir"] = self.config.prompt_data_dir
        raw_data = load_dataset(self.config.prompt_data, split=self.config.train_split, **load_kwargs)
        prompt_records = build_prompt_records(
            raw_data, self.tokenizer, self.config.source_max_length,
            self.config.apply_chat_template, self.config.prompt_data,
            is_instruct_model=self.config._is_instruct,
        )
        self.train_raw_prompts = [record["raw_prompt"] for record in prompt_records]
        self.train_prompts = [record["processed_prompt"] for record in prompt_records]
        self.train_score_prompts = [record["score_prompt"] for record in prompt_records]

        #builds an index list that is used to control which prompts are sampled next
        self._prompt_order = list(range(len(self.train_prompts))) 
        self._prompt_ptr = 0

        raw_count = len(raw_data)
        filtered_count = len(self.train_prompts)
        print(f"[Dataset] Raw: {raw_count} | Filtered: {filtered_count} ")

        self.metrics = {
            "iteration": [],
            "oracle_queries_total": [],
            "num_oracle_labels": [],
            "num_model_labels": [],
            "preference_samples": [],
            "rm_updates": [],
            "rm_buffer_size": [],
            "rm_oracle_acc": [],
            "synthetic_audit_queries_total": [],
            "num_synthetic_audited": [],
            "num_synthetic_audit_matches": [],
            "synthetic_oracle_agreement": [],
            "synthetic_oracle_agreement_cumulative": [],
            "dpo_loss": [],
            "dpo_reward_margin": [],
            "eval_win_rate": [],
            "eval_soft_win_rate": [],
            "eval/response_tok_len": [],
        }
        self.oracle_queries_used = 0
        self.synthetic_audit_queries_used = 0
        self.synthetic_audit_matches = 0

        # AERO-specific diagnostics (separate from main metrics)
        self.aero_diagnostics = []

    def train(self):
        # Set seeds for reproducibility
        self._set_seed(self.config.seed)

        total_iterations = self.config.max_iterations

        if self.config.use_online_dpo:
            explorer_name = "Online DPO (vanilla)"
            self._aero_dynamic_oracle = False
        elif self.config.use_copo:
            explorer_name = f"CoPO ({self.config.copo_mode})"
            self._aero_dynamic_oracle = False
        elif self.config.use_aero_fixed_quantile:
            explorer_name = (
                "AERO fixed quantile "
                f"(top {self.config.aero_fixed_quantile_ratio:.0%} by vote entropy)"
            )
            self._aero_dynamic_oracle = False
        elif self.config.use_aero_variant2:
            explorer_name = (
                f"AERO Variant 2 (random {self.config.aero_variant2_oracle_ratio:.0%} oracle routing)"
            )
            self._aero_dynamic_oracle = False
        elif self.config.use_aero_variant3:
            explorer_name = "AERO Variant 3 (SEA-style synthetic + aleatoric)"
            self._aero_dynamic_oracle = True
        elif self.config.use_aero_variant4:
            explorer_name = "AERO Variant 4 (confidence-gated synthetic labeling)"
            self._aero_dynamic_oracle = True
        elif self.config.use_aero_explorer:
            explorer_name = "AERO"
            self._aero_dynamic_oracle = True
        else:
            explorer_name = "SEA (Random + TS)"
            self._aero_dynamic_oracle = False

        print(f"\n[Training Plan]")
        print(f"  Explorer: {explorer_name}")
        if self.config.use_copo:
            if self.config.copo_mode == "copo_fixed":
                print("  Burn-in: none (fixed CoPO)")
                print(f"  CoPO stage 1 prompts: {self.config.copo_stage1_num_prompts}")
                print("  CoPO CFN: one-shot stage-1 pretrain, then frozen during RL")
            else:
                print(f"  Burn-in: {self.config.burn_in_period} iterations")
                print("  CoPO burn-in: 2 responses, oracle pairs only, no ERM selection, no count bonus")
                print(f"  CoPO rolling CFN buffer: {self.config.copo_cfn_buffer_maxlen}")
        else:
            print(f"  Burn-in: {self.config.burn_in_period} iterations")
        print(f"  Total iterations: {total_iterations}")
        print(f"  Rollouts: vLLM (loads from config.current_model_path)")
        print(f"  RM: persistent PairRM+Ensemble")
        print(f"  DPO: persistent HF model+optimizer")
        print(f"{'='*60}")

        # Store total_iterations for scheduler initialization
        self.total_iterations = total_iterations

        # DEBUG: Check baseline win rate before any training
        print(f"\n[DEBUG] Baseline evaluation (before training)...")
        self._run_baseline_eval()

        if self.copo_stage1_trainer is not None:
            self.copo_stage1_trainer.run(self.train_prompts, total_iterations)

        for iteration in range(total_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{total_iterations}")
            print(f"{'='*60}")

            self.run_iteration(iteration, total_iterations)

            if (iteration + 1) % self.config.eval_every_n_iters == 0:
                self.evaluate(iteration + 1)

            self.save_metrics(iteration + 1)


        print(f"\n{'='*60}\nFinal Evaluation\n{'='*60}")
        self.evaluate(iteration + 1, final=True)
        self.save_final_results()

    def run_iteration(self, iteration: int, total_iterations: int):
        raw_prompts, prompts, score_prompts = self._get_prompts_for_iteration(iteration)
        has_copo_cfn_stage = self.config.use_copo and self.config.copo_mode == "copo_rolling"
        total_stages = 5 if has_copo_cfn_stage else 4

        #1/4: vLLM rollouts
        print(f"\n[1/{total_stages}] Generation ({len(prompts)} prompts)")
        print(f"  Loading model from: {self.config.current_model_path}")
        rollout_num_samples = None
        if self.config.use_copo:
            if self.config.copo_mode == "copo_rolling" and iteration < int(self.config.burn_in_period):
                rollout_num_samples = int(self.config.copo_burnin_num_samples)
                print(f"  CoPO burn-in rollout samples per prompt: {rollout_num_samples}")
            else:
                rollout_num_samples = int(self.config.copo_num_samples)
                print(f"  CoPO rollout samples per prompt: {rollout_num_samples}")
        elif self.config.use_online_dpo:
            rollout_num_samples = 2
            print(f"  Online DPO rollout samples per prompt: {rollout_num_samples}")

        actor = Actor(
            self.config,
            eval_mode=False,
            num_samples_override=rollout_num_samples,
        )  # vLLM actor reads current_model_path
        responses = actor.generate(prompts)
        actor.cleanup()
        del actor
        self._clean_memory(light=False)

        # 2/4: collect preferences
        print(f"[2/{total_stages}] Data Collection")
        if self.config.use_online_dpo:
            explorer = OnlineDPOExplorer(self.config, step=iteration)
            print("  Using Online DPO (vanilla, no ensemble)")
        elif self.config.use_copo:
            explorer = CoPOExplorer(rm=self.rm, config=self.config, step=iteration)
            print("  Using CoPO Explorer")
        elif self.config.use_aero_fixed_quantile:
            explorer = AeroFixedQuantileExplorer(
                self.rm, self.config, step=iteration
            )
            print(
                "  Using AERO fixed-quantile Explorer "
                f"(top {self.config.aero_fixed_quantile_ratio:.0%} by vote entropy)"
            )
        elif self.config.use_aero_variant2:
            explorer = AeroVariant2Explorer(self.rm, self.config, step=iteration)
            print(
                "  Using AERO Variant 2 Explorer "
                f"(random {self.config.aero_variant2_oracle_ratio:.0%} oracle routing)"
            )
        elif self.config.use_aero_variant3:
            explorer = AeroVariant3Explorer(self.rm, self.config, step=iteration)
            print("  Using AERO Variant 3 Explorer (SEA-style synthetic)")
        elif self.config.use_aero_variant4:
            explorer = Explorer(self.rm, self.config, step=iteration)
            print("  Using AERO Variant 4 Explorer (confidence gate enabled)")
        elif self.config.use_aero_explorer:
            explorer = Explorer(self.rm, self.config, step=iteration)
            print("  Using AERO Explorer")
        else:
            explorer = SEAExplorer(self.rm, self.config, step=iteration)
            print("  Using SEA Explorer")

        policy_data, pref_stats = self.collect_preferences(
            raw_prompts=raw_prompts,
            prompts=prompts,
            score_prompts=score_prompts,
            responses=responses,
            rm=self.rm,
            explorer=explorer,
            oracle=self.oracle,
        )

        num_synthetic_audited = int(pref_stats.get("num_synthetic_audited", 0))
        num_synthetic_audit_matches = int(
            pref_stats.get("num_synthetic_audit_matches", 0)
        )
        if self.config.audit_synthetic_labels:
            self.synthetic_audit_queries_used += num_synthetic_audited
            self.synthetic_audit_matches += num_synthetic_audit_matches

        # Collect explorer diagnostics (AERO or CoPO)
        if hasattr(explorer, "get_diagnostics") and (
            self.config.use_aero_explorer
            or self.config.use_copo
            or self.config.use_aero_variant2
            or self.config.use_aero_variant3
            or self.config.use_aero_variant4
            or self.config.use_aero_fixed_quantile
        ):
            diag = explorer.get_diagnostics()
            diag["iteration"] = iteration + 1
            diag["oracle_queries_total"] = self.oracle_queries_used + pref_stats.get("num_oracle_labels", 0)
            if self.config.audit_synthetic_labels:
                diag.update(
                    {
                        "synthetic_audit_queries_total": self.synthetic_audit_queries_used,
                        "synthetic_audit_matches_total": self.synthetic_audit_matches,
                        "num_synthetic_audited": num_synthetic_audited,
                        "synthetic_oracle_agreement": pref_stats.get("synthetic_oracle_agreement"),
                        "synthetic_oracle_agreement_cumulative": (
                            self.synthetic_audit_matches
                            / self.synthetic_audit_queries_used
                            if self.synthetic_audit_queries_used > 0
                            else None
                        ),
                    }
                )
            self.aero_diagnostics.append(diag)
            self._save_aero_diagnostics()

        del explorer
        self._clean_memory(light=True)

        num_oracle_labels = int(pref_stats.get("num_oracle_labels", 0))
        num_model_labels = int(pref_stats.get("num_model_labels", 0))

        self.oracle_queries_used += num_oracle_labels

        # 3/5: train/update CoPO CFN buffer
        if has_copo_cfn_stage:
            cfn_examples = pref_stats.get("cfn_training_examples", [])
            if cfn_examples:
                print(
                    f"[3/{total_stages}] CoPO CFN Training "
                    f"({len(cfn_examples)} new responses, buffer->{min(len(self.copo_cfn_buffer) + len(cfn_examples), self.copo_cfn_buffer.maxlen)})"
                )
                self.copo_cfn_buffer.extend(cfn_examples)
                self.copo_cfn_trainer.fit_buffer(
                    list(self.copo_cfn_buffer),
                    total_iterations=total_iterations,
                )
            else:
                print(f"[3/{total_stages}] CoPO CFN Training — skipped (no buffered responses)")

        ensemble_stage = 4 if has_copo_cfn_stage else 3
        dpo_stage = 5 if has_copo_cfn_stage else 4

        # train ensemble (skip entirely for online DPO — no ensemble needed)
        num_erm_samples = len(pref_stats["oracle_data"])
        rm_stats = {"rm_updates": 0, "rm_skips": 0, "rm_skip_reason": "", "buffer_size": int(self.global_buffer.size)}

        if self.config.use_online_dpo:
            print(f"[{ensemble_stage}/{total_stages}] Ensemble Training — skipped (online DPO, no ensemble)")
        elif num_erm_samples > 0:
            print(f"[{ensemble_stage}/{total_stages}] Ensemble Training ({num_erm_samples} oracle samples)")
            oracle_dataset = TextPreferenceDataset(pref_stats["oracle_data"])
            oracle_loader = DataLoader(
                oracle_dataset,
                batch_size=32,
                shuffle=True,
                collate_fn=TextPreferenceDataset.collate_fn,
                drop_last=False,
            )
            rm_stats = self.ensemble_trainer.train_ensemble(
                rm=self.rm,
                dataloader=oracle_loader,
                buffer=self.global_buffer,
                train_bs=self.config.rm_train_bs,
                save_path=os.path.join(self.config.output_dir, "ensemble_mlp.pt"),
            )
        else:
            print("  [Ensemble] Skipping RM update: no oracle labels")

        self._clean_memory(light=False)

        print(f"[{dpo_stage}/{total_stages}] DPO Training ({len(policy_data)} samples)")

        if len(policy_data) > 0:
            self.dpo_trainer.add_data(policy_data)
            ref_swap_done = False
            oracle_offload_done = False
            dpo_error = None
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            swap_ref_to_oracle_gpu = (
                self.config.swap_oracle_ref_during_dpo
                and not self.config.use_lora  # LoRA: no separate ref model to swap
                and n_gpus == 2  # Only swap with exactly 2 GPUs
                and self.config.device_oracle != self.config.device_actor
            )
            offload_oracle_for_single_gpu = (
                self.config.swap_oracle_ref_during_dpo
                and n_gpus == 1
                and self.config.device_oracle == self.config.device_actor
            )
            if offload_oracle_for_single_gpu:
                self.oracle.move_to("cpu")
                oracle_offload_done = True
                self._clean_memory(light=False)
            if swap_ref_to_oracle_gpu:
                self.oracle.move_to("cpu")
                self.dpo_trainer.set_ref_device(self.config.device_oracle)
                oracle_offload_done = True
                ref_swap_done = True
                self._clean_memory(light=False)
            try:
                dpo_metrics = self.dpo_trainer.train(iteration + 1, total_iterations)
            except Exception as err:
                dpo_error = err
                raise
            finally:
                if ref_swap_done:
                    self.dpo_trainer.set_ref_device(self.config.device_actor)
                if oracle_offload_done:
                    self._clean_memory(light=False)
                    try:
                        self.oracle.move_to(self.config.device_oracle)
                    except Exception as move_err:
                        if dpo_error is not None:
                            print(f"[DPO] Warning: failed to move oracle back after DPO exception: {move_err}")
                        else:
                            raise
                    else:
                        self._clean_memory(light=False)

            # Save checkpoint + update rollout model path for next iteration
            ckpt_path = self.dpo_trainer.save_checkpoint(self.config.save_dir, iteration + 1)
            self.config.current_model_path = ckpt_path
            print(f"Updated rollout model path: {self.config.current_model_path}")
        else:
            dpo_metrics = {}

        # Log
        self.metrics["iteration"].append(iteration + 1)
        self.metrics["oracle_queries_total"].append(self.oracle_queries_used)
        self.metrics["num_oracle_labels"].append(num_oracle_labels)
        self.metrics["num_model_labels"].append(num_model_labels)
        self.metrics["preference_samples"].append(len(policy_data))
        self.metrics["rm_updates"].append(int(rm_stats.get("rm_updates", 0)))
        self.metrics["rm_buffer_size"].append(int(rm_stats.get("buffer_size", 0)))
        self.metrics["rm_oracle_acc"].append(float(pref_stats.get("rm_oracle_acc", 0.0)))
        self.metrics["synthetic_audit_queries_total"].append(self.synthetic_audit_queries_used)
        self.metrics["num_synthetic_audited"].append(num_synthetic_audited)
        self.metrics["num_synthetic_audit_matches"].append(
            num_synthetic_audit_matches
        )
        self.metrics["synthetic_oracle_agreement"].append(
            pref_stats.get("synthetic_oracle_agreement")
        )
        self.metrics["synthetic_oracle_agreement_cumulative"].append(
            self.synthetic_audit_matches / self.synthetic_audit_queries_used
            if self.synthetic_audit_queries_used > 0
            else None
        )
        self.metrics["dpo_loss"].append(dpo_metrics.get("dpo/loss", 0.0))
        self.metrics["dpo_reward_margin"].append(dpo_metrics.get("dpo/reward_margin", 0.0))

        print(f"\n[Summary]")
        print(f"  Oracle labels (this iter): {num_oracle_labels}")
        print(f"  Model labels (this iter):  {num_model_labels}")
        print(f"  RM updates (this iter):    {int(rm_stats.get('rm_updates', 0))}")
        print(f"  RM buffer size:            {int(rm_stats.get('buffer_size', 0))}")
        if rm_stats.get("rm_skips", 0) > 0 and rm_stats.get("rm_skip_reason"):
            print(
                f"  RM updates skipped:        {int(rm_stats.get('rm_skips', 0))} "
                f"({rm_stats.get('rm_skip_reason')})"
            )
        print(f"  Oracle queries (total): {self.oracle_queries_used}")
        rm_oracle_acc = pref_stats.get("rm_oracle_acc", float("nan"))
        if rm_oracle_acc == rm_oracle_acc:  # check not NaN
            print(f"  RM accuracy (oracle-labeled): {rm_oracle_acc:.4f}")
        if self.config.audit_synthetic_labels:
            synthetic_agreement = pref_stats.get("synthetic_oracle_agreement")
            if synthetic_agreement is None:
                print("  Synthetic-label audit: no ERM-routed pairs this iteration")
            else:
                print(
                    f"  Synthetic-label audit: {synthetic_agreement:.4f} agreement "
                    f"over {num_synthetic_audited} pairs "
                    f"(cumulative: {self.synthetic_audit_matches}/"
                    f"{self.synthetic_audit_queries_used} = "
                    f"{self.synthetic_audit_matches / self.synthetic_audit_queries_used:.4f})"
                )
        if dpo_metrics:
            print(f"  DPO loss:   {dpo_metrics.get('dpo/loss', 0.0):.4f}")
            print(f"  DPO margin: {dpo_metrics.get('dpo/reward_margin', 0.0):.4f}")

        # Free CUDA cache after DPO so next iteration's vLLM has maximum memory
        self._clean_memory(light=False)

    def collect_preferences(
        self,
        raw_prompts: List[str],
        prompts: List[str],
        score_prompts,
        responses: Dict[int, List[str]],
        rm: RewardModel,
        explorer: Union[
            SEAExplorer,
            Explorer,
            OnlineDPOExplorer,
            CoPOExplorer,
            AeroVariant2Explorer,
            AeroVariant3Explorer,
            AeroFixedQuantileExplorer,
        ],
        oracle: Oracle,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        explorer handles all selection logic:

        Oracle labels are used for:
        1. Training the ensemble reward model
        2. Training the policy (DPO)

        Ensemble/synthetic labels are used for:
        1. Training the policy (DPO) only - NOT for ensemble training
        """
        # Online DPO skips reward model entirely
        if isinstance(explorer, OnlineDPOExplorer):
            rewards = None
            oracle_candidates, synthetic_candidates = explorer.select_candidates(
                None, prompts, responses
            )
        elif isinstance(explorer, CoPOExplorer) and explorer.is_burn_in:
            rewards = None
            oracle_candidates, synthetic_candidates = explorer.select_candidates(
                None, prompts, responses
            )
        else:
            rewards = rm.get_rewards(score_prompts, responses)
            oracle_candidates, synthetic_candidates = explorer.select_candidates(
                rewards, prompts, responses
            )

        num_oracle = len(oracle_candidates.prompts)
        num_synthetic = len(synthetic_candidates.prompts) if synthetic_candidates else 0

        if num_oracle == 0 and num_synthetic == 0:
            stats = {
                "num_oracle_labels": 0,
                "num_model_labels": 0,
                "oracle_data": [],
                "rm_oracle_acc": float("nan"),
                "num_synthetic_audited": 0,
                "num_synthetic_audit_matches": 0,
                "synthetic_oracle_agreement": None,
            }
            return [], stats

        # Query oracle for oracle-labeled samples
        oracle_probs = None
        if num_oracle > 0:
            oracle_res_a = [oracle_candidates.candidates[i][0] for i in range(num_oracle)]
            oracle_res_b = [oracle_candidates.candidates[i][1] for i in range(num_oracle)]
            oracle_reward_prompts = [
                score_prompts[orig_idx]
                for orig_idx, _, _ in oracle_candidates.meta
            ]
            oracle_probs = oracle.compare(
                oracle_reward_prompts,
                oracle_res_a,
                oracle_res_b,
                batch_size=self.config.oracle_batch_size,
            )

        # Compute RM accuracy on oracle-labeled pairs (skip for online DPO)
        rm_oracle_acc = float("nan")
        if num_oracle > 0 and oracle_probs is not None and rewards is not None:
            mean_rewards = rewards.mean(0).squeeze(-1)  # (M, N)
            rm_pred = []
            oracle_labels = []
            for i, (orig_idx, anchor_idx, challenger_idx) in enumerate(oracle_candidates.meta):
                rm_pred.append(mean_rewards[orig_idx, anchor_idx] > mean_rewards[orig_idx, challenger_idx])
                oracle_labels.append(oracle_probs[i] > 0.5)
            rm_pred = torch.tensor(rm_pred, device=self.device)
            oracle_labels = torch.tensor(oracle_labels, device=self.device)
            rm_oracle_acc = (rm_pred == oracle_labels).float().mean().item()

        # Optional counterfactual audit. The oracle labels are used only for
        # this agreement statistic and never enter ERM or DPO training.
        num_synthetic_audited = 0
        num_synthetic_audit_matches = 0
        synthetic_oracle_agreement = None
        if self.config.audit_synthetic_labels:
            if type(explorer) is not Explorer:
                raise RuntimeError(
                    "Synthetic-label auditing is supported only for the standard AERO explorer"
                )
            if synthetic_candidates is not None and num_synthetic > 0:
                if len(synthetic_candidates.meta) != num_synthetic:
                    raise RuntimeError(
                        "AERO synthetic candidate metadata is required for oracle auditing"
                    )
                synthetic_audit_prompts = [
                    score_prompts[orig_idx]
                    for orig_idx, _, _ in synthetic_candidates.meta
                ]
                synthetic_res_a = [
                    synthetic_candidates.candidates[i][0]
                    for i in range(num_synthetic)
                ]
                synthetic_res_b = [
                    synthetic_candidates.candidates[i][1]
                    for i in range(num_synthetic)
                ]
                synthetic_audit_probs = oracle.compare(
                    synthetic_audit_prompts,
                    synthetic_res_a,
                    synthetic_res_b,
                    batch_size=self.config.oracle_batch_size,
                )
                erm_labels = synthetic_candidates.probs.detach().cpu() > 0.5
                oracle_labels = synthetic_audit_probs.detach().cpu() > 0.5
                num_synthetic_audited = num_synthetic
                num_synthetic_audit_matches = int(
                    (erm_labels == oracle_labels).sum().item()
                )
                synthetic_oracle_agreement = (
                    num_synthetic_audit_matches / num_synthetic_audited
                )

        # Pack oracle data for ensemble RM training
        oracle_data = []
        if num_oracle > 0:
            oracle_reward_prompts = [
                score_prompts[orig_idx]
                for orig_idx, _, _ in oracle_candidates.meta
            ]
            oracle_data = self.pack_experience(
                oracle_reward_prompts,
                oracle_candidates.candidates,
                oracle_probs,
            )

        # Pack synthetic data for DPO (already has ensemble labels)
        synthetic_data = []
        if synthetic_candidates is not None and num_synthetic > 0:
            synthetic_data = self.pack_experience(
                synthetic_candidates.prompts,
                synthetic_candidates.candidates,
                synthetic_candidates.probs,
            )

        # Combine oracle + synthetic for DPO policy training
        oracle_data_for_dpo = []
        if num_oracle > 0:
            oracle_policy_prompts = [
                prompts[orig_idx] for orig_idx, _, _ in oracle_candidates.meta
            ]
            reference_responses = None
            if isinstance(explorer, CoPOExplorer) and oracle_candidates.reference_responses:
                assert len(oracle_candidates.reference_responses) == num_oracle
                reference_responses = oracle_candidates.reference_responses
            oracle_data_for_dpo = self.pack_experience(
                oracle_policy_prompts,
                oracle_candidates.candidates,
                oracle_probs,
                reference_responses=reference_responses,
            )

        policy_data = oracle_data_for_dpo + synthetic_data
        cfn_training_examples = []
        if isinstance(explorer, CoPOExplorer):
            cfn_training_examples = explorer.build_cfn_training_examples(
                prompts=prompts,
                responses=responses,
                oracle_candidates=oracle_candidates,
            )

        stats = {
            "num_oracle_labels": num_oracle,
            "num_model_labels": num_synthetic,
            "oracle_data": oracle_data,
            "rm_oracle_acc": rm_oracle_acc,
            "cfn_training_examples": cfn_training_examples,
            "num_synthetic_audited": num_synthetic_audited,
            "num_synthetic_audit_matches": num_synthetic_audit_matches,
            "synthetic_oracle_agreement": synthetic_oracle_agreement,
        }
        return policy_data, stats

    def pack_experience(
        self,
        prompts,
        candidates,
        probs,
        reference_responses: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Pack preference data for training.

        Use DETERMINISTIC labeling
        - If P(A > B) > 0.5: A wins (chosen_idx = 0)
        - If P(A > B) <= 0.5: B wins (chosen_idx = 1)

        This is critical! Stochastic labeling adds too much noise.

        If reference_responses is provided (CoPO), each prompt also carries a
        third response used only for the exploration term during DPO.
        """
        data = []
        probs = probs.to("cpu")
        if reference_responses is not None:
            assert len(reference_responses) == len(prompts)

        # DETERMINISTIC labeling (NOT Bernoulli sampling)
        did_A_win = (probs > 0.5).long().detach()
        chosen_indices = 1 - did_A_win  # 0 if A wins (probs > 0.5), 1 if B wins

        for i in range(len(prompts)):
            winner_idx = chosen_indices[i].item()
            loser_idx = 1 - winner_idx

            chosen_text = candidates[i][winner_idx]
            rejected_text = candidates[i][loser_idx]

            same_response = chosen_text == rejected_text
            loss_mask = not same_response

            entry = {
                "prompt": prompts[i],
                "chosen": chosen_text,
                "rejected": rejected_text,
                "loss_mask": loss_mask,
            }

            if reference_responses is not None:
                entry["reference_response"] = reference_responses[i]
                entry["has_reference_response"] = True

            data.append(entry)
        return data

    def evaluate(self, iteration: int, final: bool = False):
        print(f"\n[Evaluation - Iteration {iteration}{' (final)' if final else ''}]")

        if not hasattr(self, "evaluator"):
            self.evaluator = Evaluate(self.config, self.tokenizer)

        # Cache references first (only happens once)
        self.evaluator.ensure_reference_responses_dataset()

        actor = Actor(self.config, eval_mode=True)
        eval_metrics = self.evaluator.evaluate(actor, self.oracle, step=iteration)

        self.metrics["eval_win_rate"].append(eval_metrics["win_rate"])
        self.metrics["eval_soft_win_rate"].append(eval_metrics["soft_win_rate"])
        self.metrics["eval/response_tok_len"].append(eval_metrics["eval/response_tok_len"])

        # Primary evaluation (vs SFT baseline or dataset reference)
        ref_type = "SFT baseline" if self.evaluator._needs_sft_generation else "dataset reference"
        print(f"  Win Rate (vs {ref_type}): {eval_metrics['win_rate']:.2%}")
        print(f"  Soft Win Rate:     {eval_metrics['soft_win_rate']:.4f}")

        print(f"  Response Tok Len:  {eval_metrics['eval/response_tok_len']:.1f}")
        print(f"  Eval Time:         {eval_metrics['eval/elapse']:.1f}s")

        actor.cleanup()
        del actor
        self._clean_memory(light=False)

    def _run_baseline_eval(self):
        print("[Baseline] Running SFT baseline eval using vLLM...")

        if not hasattr(self, "evaluator"):
            self.evaluator = Evaluate(self.config, self.tokenizer)

        old_path = self.config.current_model_path
        self.config.current_model_path = self.config.pretrain_model
        try:
            # cache refs FIRST (no policy Actor exists yet)
            self.evaluator.ensure_reference_responses_dataset()

            actor = Actor(self.config, eval_mode=True)

            # Generate SFT baseline responses if needed (for Llama models)
            # This must happen BEFORE training starts
            if self.evaluator._needs_sft_generation:
                self.evaluator.generate_sft_baseline(actor)
                print("[Baseline] SFT baseline generated and cached for future evaluations")
                # For SFT-vs-SFT, win rate should be ~50%
                print("[Baseline] Note: Primary eval is now model vs SFT baseline (should be ~50% initially)")

            eval_metrics = self.evaluator.evaluate(actor, self.oracle, step=0)

            ref_type = "SFT baseline" if self.evaluator._needs_sft_generation else "dataset reference"
            print(f"[Baseline] Win Rate (SFT vs {ref_type}): {eval_metrics['win_rate']:.2%}")
            print(f"[Baseline] Soft Win Rate: {eval_metrics['soft_win_rate']:.4f}")

            print(f"[Baseline] Response Tok Len: {eval_metrics['eval/response_tok_len']:.1f}")
            print(f"[Baseline] Eval Time: {eval_metrics['eval/elapse']:.1f}s")

            actor.cleanup()
            del actor
            self._clean_memory(light=False)
        finally:
            self.config.current_model_path = old_path

    def save_metrics(self, iteration: int):
        checkpoint = {
            "iteration": iteration,
            "oracle_queries_used": self.oracle_queries_used,
            "metrics": self.metrics,
        }
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, f"metrics_iter{iteration}.json")
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _save_aero_diagnostics(self):
        """Save AERO-specific diagnostics to a separate JSON file."""
        if not self.aero_diagnostics:
            return
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, "aero_diagnostics.json")
        with open(path, "w") as f:
            json.dump(self.aero_diagnostics, f, indent=2)

    def save_final_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine method name
        if self.config.use_online_dpo:
            method_name = "OnlineDPO"
        elif self.config.use_copo:
            method_name = "CoPO"
        elif self.config.use_aero_fixed_quantile:
            method_name = "AEROFixedQuantile"
        elif self.config.use_aero_variant2:
            method_name = "AEROVariant2"
        elif self.config.use_aero_variant3:
            method_name = "AEROVariant3"
        elif self.config.use_aero_variant4:
            method_name = "AEROVariant4"
        elif self.config.use_aero_explorer:
            method_name = "AERO"
        else:
            method_name = "SEA"

        # Base config
        config_dict = {
            "max_iterations": self.config.max_iterations,
            "burn_in_period": self.config.burn_in_period,
            "num_ensemble": self.config.num_ensemble,
            "train_batch": self.config.train_batch,
            "max_model_data_ratio": self.config.max_model_data_ratio,
        }

        # Add AERO-specific config if applicable
        if self.config.use_aero_explorer or self.config.use_aero_variant4:
            config_dict.update({
                "confidence_threshold": self.config.confidence_threshold,
                "confidence_filter": self.config.confidence_filter,
                "audit_synthetic_labels": self.config.audit_synthetic_labels,
            })
        if self.config.use_aero_variant2:
            config_dict["aero_variant2_oracle_ratio"] = self.config.aero_variant2_oracle_ratio
        if self.config.use_aero_fixed_quantile:
            config_dict["aero_fixed_quantile_ratio"] = (
                self.config.aero_fixed_quantile_ratio
            )
        if self.config.use_copo:
            config_dict.update({
                "copo_mode": self.config.copo_mode,
                "copo_num_samples": self.config.copo_num_samples,
            })
            if self.config.copo_mode == "copo_fixed":
                config_dict["copo_stage1_num_prompts"] = self.config.copo_stage1_num_prompts
            else:
                config_dict["copo_cfn_buffer_maxlen"] = self.config.copo_cfn_buffer_maxlen

        results = {
            "method": method_name,
            "config": config_dict,
            "metrics": self.metrics,
            "final_oracle_queries": self.oracle_queries_used,
            "final_synthetic_audit_queries": self.synthetic_audit_queries_used,
            "final_synthetic_audit_matches": self.synthetic_audit_matches,
            "final_synthetic_oracle_agreement": (
                self.synthetic_audit_matches / self.synthetic_audit_queries_used
                if self.synthetic_audit_queries_used > 0
                else None
            ),
            "total_iterations": len(self.metrics["iteration"]),
            "final_model_path": self.config.current_model_path,
        }

        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, f"results_{method_name.lower()}_{timestamp}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {path}")

    def _get_prompts_for_iteration(self, iteration: int) -> Tuple[List[str], List[str], List[Any]]:
        """
        Get prompts for this iteration.

        Every method uses ``train_batch`` prompts per iteration. The selected
        explorer determines how their responses are labeled.
        """
        batch_size = self.config.train_batch

        if self._prompt_ptr + batch_size > len(self._prompt_order):
            self._shuffle_prompts()

        idxs = self._prompt_order[self._prompt_ptr : self._prompt_ptr + batch_size]
        self._prompt_ptr += batch_size
        return (
            [self.train_raw_prompts[i] for i in idxs],
            [self.train_prompts[i] for i in idxs],
            [self.train_score_prompts[i] for i in idxs],
        )

    def _shuffle_prompts(self):
        """Shuffle prompt order between epochs"""
        if not hasattr(self, "_prompt_order"):
            return
        random.shuffle(self._prompt_order)
        self._prompt_ptr = 0

    def _clean_memory(self, light: bool = False):
        gc.collect()
        if torch.cuda.is_available() and not light:
            torch.cuda.empty_cache()

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._shuffle_prompts()
