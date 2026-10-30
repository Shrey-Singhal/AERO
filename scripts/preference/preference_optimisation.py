"""
What this does:
1. Model loaded ONCE, kept in GPU memory forever
2. Optimizer created ONCE, momentum accumulates across iterations
3. Small buffer (deque) that gets replaced each iteration
4. Train 1 full epoch on buffer each iteration
5. Cosine LR scheduler with warmup
"""

from __future__ import annotations

import math
import os
import shutil
from collections import deque
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model

from configs.config import AeroConfig
from ..reward.reward_model import RewardModel
from .preference_learning import RewardBatch, UniformBuffer
from .preference_dataset import PreferenceDataset

# Default path for ensemble checkpoint
ENSEMBLE_CKPT_PATH = "./outputs/ensemble_mlp.pt"


def _trainable_param_dtype_counts(model) -> dict[str, int]:
    dtype_counts: dict[str, int] = {}
    for param in model.parameters():
        if param.requires_grad:
            dtype_counts[str(param.dtype)] = dtype_counts.get(str(param.dtype), 0) + param.numel()
    return dtype_counts


def _keep_trainable_params_fp32(model) -> dict[str, int]:
    """
    Keep LoRA weights in fp32 for optimizer precision while the frozen base stays bf16.
    This avoids tiny AdamW updates being rounded away in bf16.
    """
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()
    return _trainable_param_dtype_counts(model)

class PersistentDPOTrainer:
    """
    Online DPO trainer that preserves training state across iterations.

    The policy, fixed reference model, optimizer, and scheduler are initialized
    once. Each iteration replaces the policy buffer with newly collected
    preferences and trains on that batch for one epoch.
    """

    def __init__(self, config: AeroConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(config.device_actor)

        self.pi_buffer = deque(maxlen=int(config.pi_buffer_maxlen_per_device))

        self.beta = float(config.algo_beta) #temperature for dpo that controls how much we drift from reference
        self.max_epochs = max(1, int(getattr(config, "dpo_epochs_per_iter", 1)))

        # persistent state
        self._model = None
        self._ref_model = None
        self._optimizer = None
        self._scheduler = None  
        self._initialized = False

        self._global_step = 0
        self._total_training_steps = None
        self._ref_device = None
        self._use_lora_ref = False
        self._cfn = None

    def _initialize(self, total_iterations: int):
        if self._initialized:
            return

        bf16_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        if self.config.use_lora:
            # LoRA path: base model frozen in bf16 + trainable LoRA adapters.
            # Reference = same model with adapters disabled (no separate ref model).
            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "torch_dtype": bf16_dtype,
            }
            if self.config.use_flash_attention_2:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.pretrain_model, **model_kwargs,
                ).to(self.device)
            except (TypeError, ValueError) as e:
                if "attn_implementation" in str(e) or "flash" in str(e).lower():
                    print(f"[DPO] Flash attention failed ({e}), falling back to default attention")
                    model_kwargs.pop("attn_implementation", None)
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.config.pretrain_model, **model_kwargs,
                    ).to(self.device)
                else:
                    raise

            lora_config = LoraConfig(
                r=int(self.config.lora_r),
                lora_alpha=int(self.config.lora_alpha),
                lora_dropout=float(self.config.lora_dropout),
                target_modules=[m.strip() for m in self.config.lora_target_modules.split(",")],
                bias=self.config.lora_bias,
                task_type="CAUSAL_LM",
            )
            self._model = get_peft_model(self._model, lora_config)
            self._model.print_trainable_parameters()

            self._model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

            trainable_dtype_counts = _keep_trainable_params_fp32(self._model)

            self._ref_model = None
            self._ref_device = str(self.device)
            self._use_lora_ref = True
            print(f"[DPO] LoRA policy on {self.device} (ref = base without adapters)")
            print(f"[DPO] LoRA trainable parameter dtypes: {trainable_dtype_counts}")
        else:
            # Full-param path: policy in fp32 (DPO gradients need precision),
            # separate ref model in bf16 on a different GPU if available.
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.pretrain_model,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch.float32,
            ).to(self.device)

            self._model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

            ref_device = torch.device(getattr(self.config, 'device_ref', self.device))
            self._ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.pretrain_model,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=bf16_dtype,
            ).to(ref_device)
            self._ref_model.eval()
            for p in self._ref_model.parameters():
                p.requires_grad = False
            self._ref_device = str(ref_device)
            self._use_lora_ref = False
            print(f"[DPO] Reference model loaded on {self._ref_device}")

        self._load_cfn()

        # AdamW — only trainable params (LoRA: adapter weights; full: all weights)
        self._optimizer = torch.optim.AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=float(self.config.lr),
            betas=(float(self.config.adam_beta_1), float(self.config.adam_beta_2)),
            weight_decay=float(self.config.l2),
        )

        #Learning Rate scheduler

        # calculate no. of optimizer steps per iteration
        steps_per_iter = math.ceil(
            math.ceil(self.config.pi_buffer_maxlen_per_device / self.config.train_batch_size_per_device)
            / self.config.grad_accum_steps
        ) * self.max_epochs
        self._total_training_steps = max(1, int(total_iterations * steps_per_iter))

        # max_step_adjustment=0.75 makes the decay happen faster than the true training length which means
        # higher lr earlier, then it settles at the minimum earlier.
        max_step_adj = self.config.max_step_adjustment
        scheduled_steps = int(self._total_training_steps * max_step_adj)
        num_warmup_steps = int(self.config.lr_warmup_ratio * scheduled_steps)

        #create the scheduler
        self._scheduler = get_scheduler(
            name=self.config.lr_scheduler,  # "cosine_with_min_lr"
            optimizer=self._optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=scheduled_steps,
            scheduler_specific_kwargs={"min_lr": float(self.config.lr) * float(self.config.min_lr_ratio)},
        )

        self._initialized = True

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        print(f"[DPO] Initialized persistent policy. Trainable params: {trainable:,}")
        print(f"[DPO] AdamW lr={self.config.lr} betas=({self.config.adam_beta_1},{self.config.adam_beta_2}) wd={self.config.l2}")
        print(f"[DPO] Scheduler={self.config.lr_scheduler} warmup={num_warmup_steps}/{scheduled_steps} "
              f"(total_opt_steps={self._total_training_steps}, max_step_adj={max_step_adj}) "
              f"min_lr={float(self.config.lr)*float(self.config.min_lr_ratio):.2e}")

    def ensure_initialized(self, total_iterations: int) -> None:
        self._initialize(total_iterations)

    def set_ref_device(self, device: Optional[str]) -> None:
        """Set target device for reference model (move immediately if initialized)."""
        if self._initialized:
            self._move_ref_model(device)

    def _load_cfn(self) -> None:
        if not self.config.use_copo:
            self._cfn = None
            return

        cfn_path = os.path.join(self.config.output_dir, "vhead.pickle")
        if not os.path.exists(cfn_path):
            self._cfn = None
            return

        try:
            self._cfn = torch.load(cfn_path, map_location=self.device, weights_only=False)
        except TypeError:
            self._cfn = torch.load(cfn_path, map_location=self.device)

        for param in self._cfn.parameters():
            param.requires_grad = False
        self._cfn.eval()

        hidden_size = int(self._model.config.hidden_size)
        print(f"[DPO] CoPO hidden size check: model={hidden_size}, cfn={self._cfn.hidden.in_features}")
        assert hidden_size == self._cfn.hidden.in_features, (
            f"CFN hidden size mismatch: model={hidden_size}, cfn={self._cfn.hidden.in_features}"
        )
        print(f"[DPO] CoPO: loaded frozen CFN from {cfn_path}")

    def _move_ref_model(self, device: Optional[str]) -> None:
        if device is None or self._ref_model is None:
            return
        cur = str(next(self._ref_model.parameters()).device)
        if cur != device:
            self._ref_model.to(device)
        self._ref_device = device

    def add_data(self, preference_data: List[Dict[str, Any]]):
        # Replace buffer each iteration
        self.pi_buffer.clear()
        self.pi_buffer.extend(dict(item) for item in preference_data)
        print(f"[DPO] Buffer replaced: {len(self.pi_buffer)} samples")

    def train(self, iteration: int, total_iterations: int) -> Dict[str, float]:
        self._initialize(total_iterations)
        if self.config.use_copo:
            self._load_cfn()

        if len(self.pi_buffer) == 0:
            print("[DPO] Buffer empty, skipping")
            return {}

        dataset = PreferenceDataset(
            buffer=list(self.pi_buffer),
            tokenizer=self.tokenizer,
            prompt_max_length=self.config.prompt_max_length,
            generate_max_length=self.config.generate_max_length,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=int(self.config.train_batch_size_per_device),
            shuffle=True,
            drop_last=False,  # Use all samples, even if final batch is smaller
            collate_fn=dataset.collate_fn,
        )

        grad_accum = int(self.config.grad_accum_steps)

        self._model.train() #put the model in training mode
        #Clears old gradients before starting this iteration. set_to_none=True is a memory‑friendly way to clear grads.
        self._optimizer.zero_grad(set_to_none=True) 

        total_micro = 0
        total_opt = 0
        loss_sum = 0.0
        chosen_r_sum = 0.0
        rejected_r_sum = 0.0
        margin_sum = 0.0
        margin_sq_sum = 0.0
        margin_count = 0
        unmasked_sum = 0
        total_count = 0
        use_copo = getattr(self.config, 'use_copo', False)
        copo_bonus_sum = 0.0
        copo_ratio_sum = 0.0
        copo_losses_count_sum = 0.0
        copo_losses_dpo_sum = 0.0
        copo_debug_logged = False
        copo_bonus_applied = False

        for _ in range(self.max_epochs):
            for step, batch in enumerate(dataloader):
                chosen_ids, chosen_mask, rejected_ids, rejected_mask, extras = batch

                chosen_ids = chosen_ids.to(self.device)
                chosen_mask = chosen_mask.to(self.device)
                rejected_ids = rejected_ids.to(self.device)
                rejected_mask = rejected_mask.to(self.device)

                prompt_lens = extras["prompt_ids_lens"].to(self.device)     
                loss_masks = extras["loss_masks"].to(self.device)           
                # Trainable LoRA/full-model weights stay in fp32 for optimizer precision,
                # while autocast runs the forward/loss path in bf16 to save memory.
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    #forward pass to get log-probs for chosen and rejected responses.
                    pi_c, pi_r = self._forward(self._model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, prompt_lens)
                    with torch.no_grad():
                        if self._use_lora_ref:
                            with self._model.disable_adapter():
                                ref_c, ref_r = self._forward(self._model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, prompt_lens)
                        else:
                            ref_c, ref_r = self._forward(self._ref_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, prompt_lens)
                        if ref_c.device != self.device:
                            ref_c = ref_c.to(self.device)
                            ref_r = ref_r.to(self.device)

                    # DPO
                    pi_logratios = pi_c - pi_r
                    ref_logratios = ref_c - ref_r
                    logits = pi_logratios - ref_logratios

                    # keep samples where loss_mask is True
                    valid = loss_masks.float()
                    per_ex_loss = -F.logsigmoid(self.beta * logits)

                    if use_copo and "reference_ids" in extras and self._cfn is None:
                        raise FileNotFoundError("CoPO batch contains reference responses but no CFN checkpoint is loaded.")
                    apply_copo_bonus = (
                        use_copo
                        and self._cfn is not None
                        and "reference_ids" in extras
                    )
                    if apply_copo_bonus:
                        reference_ids = extras["reference_ids"].to(self.device)
                        reference_mask = extras["reference_masks"].to(self.device)
                        reference_prompt_lens = extras["reference_prompt_ids_lens"].to(self.device)

                        policy_out = self._model(
                            input_ids=reference_ids,
                            attention_mask=reference_mask,
                            output_hidden_states=True,
                        )
                        policy_response_logits = policy_out.logits

                        with torch.no_grad():
                            if self._use_lora_ref:
                                with self._model.disable_adapter():
                                    ref_out = self._model(
                                        input_ids=reference_ids,
                                        attention_mask=reference_mask,
                                        output_hidden_states=True,
                                    )
                            else:
                                ref_input_ids = reference_ids
                                ref_attention_mask = reference_mask
                                if self._ref_device != str(self.device):
                                    ref_input_ids = reference_ids.to(self._ref_device)
                                    ref_attention_mask = reference_mask.to(self._ref_device)
                                ref_out = self._ref_model(
                                    input_ids=ref_input_ids,
                                    attention_mask=ref_attention_mask,
                                    output_hidden_states=True,
                                )
                            reference_response_logits = ref_out.logits
                            reference_response_hidden = ref_out.hidden_states[-1]

                        policy_response_logps = self._sum_response_logps(
                            policy_response_logits,
                            reference_ids,
                            reference_mask,
                            reference_prompt_lens.tolist(),
                        )
                        reference_response_logps = self._sum_response_logps(
                            reference_response_logits.to(self.device),
                            reference_ids,
                            reference_mask,
                            reference_prompt_lens.tolist(),
                        )

                        copo_alpha = float(self.config.copo_alpha)
                        bonus_exponent = float(self.config.copo_bonus_exponent)

                        with torch.no_grad():
                            flips = self._cfn(reference_response_hidden.to(self.device))
                            mean_square_flips = torch.mean(torch.square(flips), dim=1, keepdim=True)
                            count_reward_raw = mean_square_flips ** bonus_exponent
                            # Keep batch dimension even for singleton micro-batches.
                            count_reward = 0.1 * count_reward_raw.squeeze(-1)

                        assert count_reward.shape == (reference_ids.shape[0],)
                        response_logratios = policy_response_logps - reference_response_logps
                        logratio_clip = torch.clip(torch.exp(response_logratios), max=5.0)
                        losses_count = -1.0 * copo_alpha * (logratio_clip * count_reward).detach() * policy_response_logps
                        losses_dpo = per_ex_loss
                        per_ex_loss = per_ex_loss + losses_count
                        copo_bonus_applied = True

                        copo_bonus_sum += float(count_reward.mean().item())
                        copo_ratio_sum += float(logratio_clip.mean().item())
                        copo_losses_count_sum += float(losses_count.mean().item())
                        copo_losses_dpo_sum += float(losses_dpo.mean().item())

                        if not copo_debug_logged:
                            print(
                                "[DPO] CoPO sanity: "
                                f"policy_logps_grad={policy_response_logps.requires_grad} "
                                f"count_reward_grad={count_reward.requires_grad} "
                                f"logratio_grad={logratio_clip.requires_grad} "
                                f"losses_count_grad={losses_count.requires_grad}"
                            )
                            print(
                                "[DPO] CoPO loss scales: "
                                f"losses_dpo={losses_dpo.mean().item():.6f} "
                                f"losses_count={losses_count.mean().item():.6f} "
                                f"count_reward={count_reward.mean().item():.6f} "
                                f"logratio_clip={logratio_clip.mean().item():.6f}"
                            )
                            copo_debug_logged = True

                    if valid.sum() > 0:
                        loss = (per_ex_loss * valid).sum() / valid.sum()
                    else:
                        loss = per_ex_loss.mean() * 0.0  # all masked

                (loss / grad_accum).backward()

                # logging
                total_micro += 1
                loss_sum += float(loss.item())
                unmasked = int(valid.sum().item())
                unmasked_sum += unmasked
                total_count += int(valid.numel())

                chosen_r_sum += float((self.beta * (pi_c - ref_c)).mean().item())
                rejected_r_sum += float((self.beta * (pi_r - ref_r)).mean().item())

                # Margin stats over unmasked samples (pi_c - pi_r)
                if unmasked > 0:
                    diffs = (pi_c - pi_r).detach()
                    diffs = diffs[loss_masks]
                    margin_sum += float(diffs.sum().item())
                    margin_sq_sum += float((diffs ** 2).sum().item())
                    margin_count += int(diffs.numel())

                # optimizer step
                if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), float(self.config.max_grad_norm))
                    self._optimizer.step()
                    self._scheduler.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    total_opt += 1
                    self._global_step += 1

        avg_loss = loss_sum / max(1, total_micro)
        avg_chosen = chosen_r_sum / max(1, total_micro)
        avg_rejected = rejected_r_sum / max(1, total_micro)
        lr_now = float(self._scheduler.get_last_lr()[0])
        frac_unmasked = float(unmasked_sum / max(1, total_count))
        too_long_count = int(getattr(dataset, "too_long_prompt_count", 0))
        total_dataset_samples = int(getattr(dataset, "total_getitem_count", 0))
        too_long_rate = too_long_count / max(1, total_dataset_samples)
        if margin_count > 0:
            margin_mean = margin_sum / margin_count
            margin_var = max(margin_sq_sum / margin_count - margin_mean ** 2, 0.0)
            margin_std = margin_var ** 0.5
        else:
            margin_mean = 0.0
            margin_std = 0.0
        eff_batch = int(self.config.train_batch_size_per_device) * grad_accum

        print(
            f"[DPO] Iter {iteration}: micro_steps={total_micro}, opt_steps={total_opt}, "
            f"eff_batch={eff_batch}, loss={avg_loss:.4f}, margin={(avg_chosen-avg_rejected):.4f}, "
            f"unmasked={frac_unmasked:.2%}, lr={lr_now:.2e}"
        )
        print(
            f"[DPO] Prompt-length diagnostic: too_long={too_long_count}/{total_dataset_samples} "
            f"({too_long_rate:.2%})"
        )
        print(
            f"[DPO] Pre-update margins (pi_c-pi_r) mean={margin_mean:.4f} std={margin_std:.4f}"
        )

        result = {
            "dpo/loss": avg_loss,
            "dpo/chosen_reward": avg_chosen,
            "dpo/rejected_reward": avg_rejected,
            "dpo/reward_margin": (avg_chosen - avg_rejected),
            "dpo/micro_steps": total_micro,
            "dpo/optimizer_steps": total_opt,
            "dpo/buffer_size": len(self.pi_buffer),
            "dpo/learning_rate": lr_now,
            "dpo/unmasked_fraction": frac_unmasked,
            "dpo/too_long_prompt_count": too_long_count,
            "dpo/too_long_prompt_rate": too_long_rate,
            "dpo/margin_mean": margin_mean,
            "dpo/margin_std": margin_std,
        }
        if use_copo and copo_bonus_applied:
            result["dpo/copo_bonus_mean"] = copo_bonus_sum / max(1, total_micro)
            result["dpo/copo_ratio_mean"] = copo_ratio_sum / max(1, total_micro)
            result["dpo/copo_losses_count_mean"] = copo_losses_count_sum / max(1, total_micro)
            result["dpo/copo_losses_dpo_mean"] = copo_losses_dpo_sum / max(1, total_micro)
            print(
                "[DPO] CoPO stats: "
                f"bonus={result['dpo/copo_bonus_mean']:.6f} "
                f"ratio={result['dpo/copo_ratio_mean']:.6f} "
                f"losses_count={result['dpo/copo_losses_count_mean']:.6f} "
                f"losses_dpo={result['dpo/copo_losses_dpo_mean']:.6f}"
            )
        return result

    def _forward(self, model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, prompt_lens):
        model_device = next(model.parameters()).device
        if chosen_ids.device != model_device:
            chosen_ids = chosen_ids.to(model_device)
            chosen_mask = chosen_mask.to(model_device)
            rejected_ids = rejected_ids.to(model_device)
            rejected_mask = rejected_mask.to(model_device)
            prompt_lens = prompt_lens.to(model_device)

        B = chosen_ids.size(0)
        max_len = max(chosen_ids.size(1), rejected_ids.size(1))
        pad_id = int(self.tokenizer.pad_token_id)

        def pad_right(x, value):
            if x.size(1) == max_len:
                return x
            return F.pad(x, (0, max_len - x.size(1)), value=value)

        input_ids = torch.cat([pad_right(chosen_ids, pad_id), pad_right(rejected_ids, pad_id)], dim=0)
        attn_mask = torch.cat([pad_right(chosen_mask.long(), 0), pad_right(rejected_mask.long(), 0)], dim=0)

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits

        prompt_lens2 = torch.cat([prompt_lens, prompt_lens], dim=0).tolist()
        logps = self._sum_response_logps(logits, input_ids, attn_mask, prompt_lens2)

        return logps[:B], logps[B:]

    @staticmethod
    def _sum_response_logps(logits, labels, attention_mask, prompt_id_lens: List[int]) -> torch.Tensor:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = attention_mask[:, 1:].bool()

        # Avoid materializing the full log_softmax tensor, which doubles peak memory.
        selected_logits = shift_logits.gather(dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)
        token_logps = selected_logits - torch.logsumexp(shift_logits, dim=-1)

        resp_mask = shift_mask.clone()
        for i, pl in enumerate(prompt_id_lens):
            cut = max(int(pl) - 1, 0)  # -1 because of shift
            cut = min(cut, resp_mask.size(1))
            resp_mask[i, :cut] = False

        return (token_logps * resp_mask.float()).sum(dim=-1)

    def get_model(self):
        # ensure initialized at least once
        if not self._initialized:
            self._initialize(total_iterations=1)
        return self._model

    def save_checkpoint(self, path: str, iteration: int) -> str:
        if not self._initialized:
            self._initialize(total_iterations=1)

        save_path = os.path.join(path, f"policy_iter{iteration}")
        os.makedirs(save_path, exist_ok=True)

        if self._use_lora_ref:
            # LoRA: save only adapter weights (adapter_config.json + adapter_model.*)
            # vLLM loads base model + adapter via LoRARequest
            self._model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        else:
            # Full-param: save weights in bf16 for vLLM compatibility.
            # vLLM auto-converts fp32 to fp16 for speed. Some model components (like rotary embedding buffers) are
            # created in bf16 which causes dtype mismatch and causes vllm to crash
            # Saving in bf16 ensures vLLM loads with the correct dtype.
            bf16_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            state_dict_bf16 = {k: v.to(bf16_dtype) for k, v in self._model.state_dict().items()}

            original_dtype = getattr(self._model.config, "torch_dtype", None)
            self._model.config.torch_dtype = bf16_dtype
            self._model.save_pretrained(save_path, state_dict=state_dict_bf16)
            self._model.config.torch_dtype = original_dtype

            del state_dict_bf16
            self.tokenizer.save_pretrained(save_path)

        # delete previous checkpoint, but keep every 100th for offline evaluation
        if iteration > 1:
            prev = iteration - 1
            if prev % 100 != 0:  # keep 100, 200, 300, 400, 500
                old = os.path.join(path, f"policy_iter{prev}")
                if os.path.exists(old):
                    shutil.rmtree(old, ignore_errors=True)

        print(f"[DPO] Saved checkpoint: {save_path}")
        return save_path


class PreferenceOptimisation:
    def __init__(self, config: AeroConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        # Use device_erm for ERM training (same device as RewardModel backbone + ensemble)
        self.device = getattr(config, 'device_erm', config.device_actor)

    def save_ensemble(self, rm: RewardModel, path: str = ENSEMBLE_CKPT_PATH):
        ckpt = {
            "state_dict": rm.model.state_dict(),
            "init_params": getattr(rm.model, "init_params", None),
        }
        torch.save(ckpt, path)
        print(f"  [Ensemble] Saved checkpoint: {path}")


    def train_ensemble(
        self,
        rm: RewardModel,
        dataloader: DataLoader,
        buffer: UniformBuffer,
        train_bs: int,
        save_path: str = ENSEMBLE_CKPT_PATH,
    ) -> Dict[str, Any]:
        """
        Train the ensemble on oracle-labeled preferences.
        """
        device = self.device

        rm.to(device)
        rm.backbone.eval()

        total_queries = int(getattr(buffer, "total_num_queries", 1))
        if total_queries < 1:
            total_queries = 1

        print("  [Ensemble] Starting training...")

        # Add ALL data to buffer first
        samples_added = 0
        for batch in dataloader:
            # TextPreferenceDataset yields raw prompt/response strings.
            if isinstance(batch, dict):
                prompts = batch.get("prompt_text") or batch.get("prompt") or batch.get("prompts")
                chosen = batch.get("chosen_text") or batch.get("chosen")
                rejected = batch.get("rejected_text") or batch.get("rejected")
            else:
                prompts = [x["prompt"] for x in batch]
                chosen = [x["chosen"] for x in batch]
                rejected = [x["rejected"] for x in batch]

            B = len(prompts)
            total_queries += B
            samples_added += B

            # tokenize pairs for PairRM backbone
            chosen_tok = [rm.backbone.tokenize_pair(p, c) for p, c in zip(prompts, chosen)]
            rej_tok = [rm.backbone.tokenize_pair(p, r) for p, r in zip(prompts, rejected)]

            chosen_enc = rm.backbone.tokenizer.pad({"input_ids": chosen_tok}, return_tensors="pt")
            rej_enc = rm.backbone.tokenizer.pad({"input_ids": rej_tok}, return_tensors="pt")

            feats_c, feats_r = [], []
            bs = int(self.config.backbone_batch)

            with torch.no_grad():
                for ndx in range(0, B, bs):
                    sl = slice(ndx, min(ndx + bs, B))
                    c_inp = {k: v[sl].to(device) for k, v in chosen_enc.items()}
                    r_inp = {k: v[sl].to(device) for k, v in rej_enc.items()}
                    feats_c.append(rm.backbone(**c_inp))
                    feats_r.append(rm.backbone(**r_inp))

            feats_c = torch.cat(feats_c, dim=0)
            feats_r = torch.cat(feats_r, dim=0)

            pair_feats = torch.stack([feats_c, feats_r], dim=1)  # (B, 2, D)
            loss_masks = torch.ones((B,), device=device, dtype=torch.float32)

            buffer.extend(
                RewardBatch(pair_features=pair_feats, loss_masks=loss_masks)
            )

        # Update total query count for regularization scaling
        buffer.total_num_queries = total_queries

        rm.model.train()
        stats = rm.learn_from_buffer(buffer, train_bs)
        rm_updated = stats.get("rm/updated", False)

        rm.model.eval()
        self.save_ensemble(rm, save_path)
        return {
            "rm_updates": 1 if rm_updated else 0,
            "rm_skips": 0 if rm_updated else 1,
            "rm_skip_reason": stats.get("rm/skip_reason", ""),
            "buffer_size": int(buffer.size),
            "samples_added": samples_added,
        }
