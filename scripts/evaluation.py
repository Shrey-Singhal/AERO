import os
import json
import time
import gc
import numpy as np
import torch
from datasets import load_dataset

from configs.config import AeroConfig
from .dataset_utils import (
    extract_prompt,
    extract_prompt_messages,
    wrap_prompt_for_instruct,
    extract_ultrafeedback_response,
)


class Evaluate:
    """
    Win-rate evaluation against dataset reference responses.

    Follows the PromptDataset pattern:
    - Stores both raw prompts and processed prompts (after chat template)
    - Filters by prompt_max_length
    - Uses eval_split and max_eval from config
    """

    def __init__(self, config: AeroConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.raw_prompts = []
        self.processed_prompts = []
        self.score_prompts = []
        self.reference_responses = []  # SFT baseline or Pythia reference
        self.sft_baseline_cache_path = os.path.join(config.output_dir, "sft_baseline_cache.json")
        self._load_eval_data()

    def _load_eval_data(self):

        split = self.config.eval_split
        load_kwargs = {}
        if self.config.prompt_data_dir:
            load_kwargs["data_dir"] = self.config.prompt_data_dir
        ds = load_dataset(self.config.prompt_data, split=split, **load_kwargs)
        columns = list(ds.column_names)
        ref_key = self.config.eval_reference_key

        # Check if we need to use cached SFT baseline
        self._needs_sft_generation = (ref_key == "sft-baseline")

        self._is_ultrafeedback = (self.config._dataset_type == "ultrafeedback")

        # For SFT baseline, references are generated after loading the prompts.
        if not self._needs_sft_generation:
            # For ultrafeedback, "chosen" is a message list, not a string column
            if self._is_ultrafeedback and ref_key == "chosen":
                if "chosen" not in columns:
                    raise ValueError(f"UltraFeedback dataset missing 'chosen' column. Available: {columns}")
            else:
                # Fallback reference keys if primary not found
                fallback_keys = ["summary", "response", "output", "answer"]
                if ref_key not in columns:
                    print(f"[Evaluate] Warning: '{ref_key}' not in dataset columns")
                    for fallback in fallback_keys:
                        if fallback in columns:
                            print(f"[Evaluate] Using fallback reference key: '{fallback}'")
                            ref_key = fallback
                            break
                    else:
                        print(f"[Evaluate] Available columns: {columns}")
                        raise ValueError(f"No valid reference key found. Tried: {[self.config.eval_reference_key] + fallback_keys}")

        max_samples = min(len(ds), self.config.max_eval)
        print(f"[Evaluate] Loading up to {max_samples} samples from {self.config.prompt_data} ({split} split)...")

        self.raw_prompts = []
        self.processed_prompts = []
        self.score_prompts = []
        self.reference_responses = []
        skipped_extract = 0
        skipped_length = 0
        skipped_reference = 0
        use_chat_template = bool(self.config.apply_chat_template)

        for i in range(max_samples):
            # Extract raw prompt
            raw_prompt, source = extract_prompt(ds[i], columns, self.config.prompt_data)
            if raw_prompt is None:
                skipped_extract += 1
                continue
            prompt_messages = extract_prompt_messages(ds[i], columns, self.config.prompt_data)
            score_prompt = prompt_messages if prompt_messages else raw_prompt

            # Wrap with instruction for instruct models (e.g., "Summarize..." for TL;DR)
            if prompt_messages:
                if use_chat_template:
                    processed_prompt = self.tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    processed_prompt = raw_prompt
            else:
                wrapped_prompt = wrap_prompt_for_instruct(
                    raw_prompt, self.config.prompt_data, self.config._is_instruct
                )

                # Apply chat template if configured
                if use_chat_template:
                    processed_prompt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": wrapped_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    processed_prompt = wrapped_prompt

            # Filter by prompt_max_length
            token_len = len(self.tokenizer(processed_prompt)["input_ids"])
            if token_len > self.config.prompt_max_length:
                skipped_length += 1
                continue

            # For SFT baseline mode, the reference is generated later.
            if self._needs_sft_generation:
                reference = ""
            elif self._is_ultrafeedback and ref_key == "chosen":
                # UltraFeedback: extract assistant response from chosen conversation
                reference = extract_ultrafeedback_response(ds[i], "chosen")
                if not reference or not reference.strip():
                    skipped_reference += 1
                    continue
            else:
                reference = ds[i].get(ref_key, "")
                if not isinstance(reference, str) or not reference.strip():
                    skipped_reference += 1
                    continue

            self.raw_prompts.append(raw_prompt)
            self.processed_prompts.append(processed_prompt)
            self.score_prompts.append(score_prompt)
            self.reference_responses.append(reference)

        print(f"[Evaluate] Loaded {len(self.raw_prompts)} prompts.")
        print(f"[Evaluate] Skipped: {skipped_extract} (extract failed), {skipped_length} (too long)")
        if self._needs_sft_generation:
            print(f"[Evaluate] Primary reference: SFT baseline (will be generated)")
        else:
            print(f"[Evaluate] Primary reference key: {ref_key} (dataset)")
        if skipped_reference:
            print(f"[Evaluate] Skipped: {skipped_reference} (missing reference)")

    def ensure_reference_responses_dataset(self):
        """Load and cache dataset references. Call BEFORE creating policy Actor."""
        if not self.raw_prompts:
            raise RuntimeError(
                "Prompts not loaded. Recreate Evaluate to reload the dataset."
            )

    def generate_sft_baseline(self, actor=None) -> bool:
        """
        Generate SFT baseline responses using the initial model (before training).
        Should be called BEFORE any training updates.

        Returns True if generation was performed, False if loaded from cache.
        """
        if not self._needs_sft_generation:
            print("[Evaluate] SFT baseline not needed (using dataset reference)")
            return False

        # Check if we have cached baseline
        if os.path.exists(self.sft_baseline_cache_path):
            print(f"[Evaluate] Loading cached SFT baseline from {self.sft_baseline_cache_path}")
            with open(self.sft_baseline_cache_path, "r") as f:
                cache = json.load(f)

            # Verify cache matches current prompts
            if len(cache.get("responses", [])) == len(self.processed_prompts):
                self.reference_responses = cache["responses"]
                print(f"[Evaluate] Loaded {len(self.reference_responses)} cached SFT baseline responses")
                return False
            else:
                print("[Evaluate] Cache size mismatch, regenerating...")

        if actor is None:
            raise RuntimeError("SFT baseline cache missing and no actor was provided for generation.")

        print(f"[Evaluate] Generating SFT baseline responses for {len(self.processed_prompts)} prompts...")
        st_time = time.time()

        # Generate using the initial model
        model_responses = actor.generate(self.processed_prompts)
        self.reference_responses = [model_responses[i][0] for i in range(len(self.processed_prompts))]

        elapsed = time.time() - st_time
        print(f"[Evaluate] Generated SFT baseline in {elapsed:.1f}s")

        # Cache the baseline responses
        cache = {
            "model": self.config.pretrain_model,
            "num_prompts": len(self.processed_prompts),
            "responses": self.reference_responses,
        }
        with open(self.sft_baseline_cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"[Evaluate] Cached SFT baseline to {self.sft_baseline_cache_path}")

        return True

    def baseline_self_eval(self) -> dict:
        """
        Fast-path metrics when the policy is evaluated against its own cached SFT
        baseline. In this case every comparison is a tie by construction.
        """
        if not self.reference_responses or not any(r.strip() for r in self.reference_responses):
            raise RuntimeError("SFT baseline responses are not loaded.")

        response_token_lens = [
            len(self.tokenizer.encode(resp, add_special_tokens=False))
            for resp in self.reference_responses
        ]
        avg_response_tok_len = np.mean(response_token_lens).item()
        return {
            "eval/rm_win_rate": 0.5,
            "eval/score": 0.5,
            "eval/eval_count": len(self.raw_prompts),
            "eval/elapse": 0.0,
            "eval/response_tok_len": float(avg_response_tok_len),
            "win_rate": 0.5,
            "soft_win_rate": 0.5,
        }

    def _release_actor_before_oracle(self, actor) -> None:
        """
        On single-GPU jobs the vLLM actor and the oracle share the same device.
        Free the actor after generation so oracle scoring does not OOM.
        """
        if not torch.cuda.is_available():
            return
        if self.config.device_actor != self.config.device_oracle:
            return

        print("[Evaluate] Releasing vLLM actor before oracle scoring on shared GPU")
        actor.cleanup()
        gc.collect()
        torch.cuda.empty_cache()

    def evaluate(self, actor, oracle, step: int = 0, save_path: str = None) -> dict:
        """
        Evaluate current policy vs cached dataset references.

        - Generate using processed_prompts (what model sees)
        - Compare using score_prompts (same structured history used for scoring)
        - Simple > 0.5 threshold for win rate
        - Handles both numpy and torch from oracle
        """
        if not self.reference_responses or not any(r.strip() for r in self.reference_responses):
            raise RuntimeError(
                "Reference responses not loaded. "
                "Call evaluator.generate_sft_baseline() or ensure references are in dataset."
            )

        st_time = time.time()

        # Generate model responses using processed prompts
        model_responses = actor.generate(self.processed_prompts)
        model_responses_flat = [model_responses[i][0] for i in range(len(self.processed_prompts))]
        self._release_actor_before_oracle(actor)

        # Model vs SFT baseline (or dataset reference)
        win_probs = oracle.compare(
            self.score_prompts,
            model_responses_flat,
            self.reference_responses,
            batch_size=self.config.eval_batch_size
        )

        # Handle both torch tensor and numpy array returns
        if isinstance(win_probs, torch.Tensor):
            win_probs_np = win_probs.detach().cpu().numpy()
        else:
            win_probs_np = np.asarray(win_probs)

        # Win rate: treat exact ties (0.5) as half-wins.
        wins = win_probs_np > 0.5
        ties = win_probs_np == 0.5
        win_rate = (wins.mean() + 0.5 * ties.mean()).item()

        # Soft win rate (mean probability)
        soft_win_rate = np.mean(win_probs_np).item()

        # Average response token length
        response_token_lens = [
            len(self.tokenizer.encode(resp, add_special_tokens=False))
            for resp in model_responses_flat
        ]
        avg_response_tok_len = np.mean(response_token_lens).item()

        elapsed = time.time() - st_time

        # Save evaluation results to JSON
        if save_path is None:
            save_path = self.config.output_dir
        eval_res_path = os.path.join(save_path, "eval_results")
        os.makedirs(eval_res_path, exist_ok=True)

        eval_results = {
            "prompts": self.raw_prompts,
            "format_prompts": self.processed_prompts,
            "outputs": model_responses_flat,
            "references_primary": self.reference_responses,
            "scores_primary": win_probs_np.tolist(),
            "wins_primary": wins.tolist(),
            "ties_primary": ties.tolist(),
        }

        with open(os.path.join(eval_res_path, f"{step}.json"), "w") as f:
            json.dump(eval_results, f, indent=2)

        # Build result dict
        result = {
            "eval/rm_win_rate": float(win_rate),
            "eval/score": float(soft_win_rate),
            "eval/eval_count": len(self.raw_prompts),
            "eval/elapse": float(elapsed),
            "eval/response_tok_len": float(avg_response_tok_len),
            # Backward-compatible keys
            "win_rate": float(win_rate),
            "soft_win_rate": float(soft_win_rate),
            "num_eval": len(self.raw_prompts),
        }

        return result
