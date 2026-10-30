from typing import Dict, List, Optional
import os
import gc
import zlib

import torch

from configs.config import AeroConfig, is_llama_model, is_instruct_model

import vllm
from vllm.lora.request import LoRARequest

_SUPPORTED_VLLM_LORA_RANKS = (8, 16, 32, 64)

def _is_lora_adapter_dir(path: str) -> bool:
    """Returns True if path looks like a PEFT LoRA adapter folder."""
    if path is None:
        return False
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))

def _get_stop_token_ids(tokenizer, model_name: str) -> Optional[List[int]]:
    """
    Get stop token IDs for different model types.

    For Llama 3 Instruct: stop at <|eot_id|> (end of turn) in addition to EOS.
    This prevents the model from generating multiple turns.
    """
    stop_ids = []

    if is_llama_model(model_name) and is_instruct_model(model_name):
        # Llama 3 Instruct special tokens
        # <|eot_id|> = 128009 (end of turn)
        # <|end_of_text|> = 128001 (EOS)
        special_tokens = ["<|eot_id|>", "<|end_of_text|>"]
        for token in special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != tokenizer.unk_token_id:
                stop_ids.append(token_id)

    # Always include EOS token if not already present
    if tokenizer.eos_token_id and tokenizer.eos_token_id not in stop_ids:
        stop_ids.append(tokenizer.eos_token_id)

    return stop_ids if stop_ids else None

def _expandable_segments_enabled() -> bool:
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    return "expandable_segments:True" in alloc_conf.replace(" ", "")

def _is_saved_policy_checkpoint_path(path: Optional[str]) -> bool:
    if not path or not os.path.exists(path):
        return False
    base = os.path.basename(os.path.normpath(path))
    return base.startswith("policy_iter")

class Actor:
    """vLLM text-generation worker for rollouts."""

    def __init__(
        self,
        config: AeroConfig,
        eval_mode: bool = False,
        num_samples_override: Optional[int] = None,
        model_path_override: Optional[str] = None,
    ):
        self.config = config
        self.device = config.device_actor
        self.eval_mode = False
        self.model_path_override = model_path_override

        self.llm = None
        self.tokenizer = None
        self.sampling_params = None
        self.vllm_lora_enabled = False
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.num_samples = config.num_samples
        self.set_generation_profile(eval_mode=eval_mode, num_samples_override=num_samples_override)
        self._init_vllm()

    def set_generation_profile(
        self,
        eval_mode: bool,
        num_samples_override: Optional[int] = None,
    ) -> None:
        self.eval_mode = eval_mode
        if eval_mode:
            self.temperature = self.config.eval_temperature
            self.max_tokens = self.config.eval_max_tokens
            default_num_samples = 1 if self.config.eval_temperature == 0.0 else self.config.num_samples
        else:
            self.temperature = self.config.temperature
            self.max_tokens = self.config.max_tokens
            default_num_samples = self.config.num_samples
        self.num_samples = (
            int(num_samples_override) if num_samples_override is not None else int(default_num_samples)
        )

        if self.llm is not None and self.tokenizer is not None:
            self.sampling_params = self._build_vllm_sampling_params()

    def _ensure_padding(self):
        """Left padding + pad token for decoder-only models."""
        assert self.tokenizer is not None
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _init_vllm(self):
        """Initialize vLLM for rollouts or evaluation."""
        cur_path = self.model_path_override or self.config.current_model_path
        print(f"[Actor] vLLM loading from: {cur_path}")

        # Adapter checkpoints use the base model for the vLLM engine and tokenizer.
        enable_lora = bool(self.config.use_lora)
        use_adapter = enable_lora and _is_lora_adapter_dir(cur_path)
        vllm_model_path = self.config.pretrain_model if use_adapter else cur_path

        # vLLM engine
        vllm_args = {
            "model": vllm_model_path,
            "tokenizer": vllm_model_path,
            "trust_remote_code": self.config.trust_remote_code,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "dtype": "bfloat16",  # Match the precision used by saved policy checkpoints.
            "disable_log_stats": True,
            "max_model_len": self.config.max_seq_len,
        }
        visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        is_single_rollout_gpu = int(self.config.tensor_parallel_size) == 1
        is_single_gpu_job = visible_gpus == 1 and is_single_rollout_gpu
        is_checkpoint_rollout = is_single_rollout_gpu and _is_saved_policy_checkpoint_path(cur_path)
        
        if (
            is_single_gpu_job
            and not self.eval_mode
            and self.num_samples > 1
        ):
            vllm_args["max_num_seqs"] = 64
       
        elif is_checkpoint_rollout:
            vllm_args["max_num_seqs"] = 64
        # Use local multiprocessing for tensor-parallel vLLM workers.
        if self.config.tensor_parallel_size > 1:
            vllm_args["distributed_executor_backend"] = "mp"
        
        if (
            self.config.tensor_parallel_size > 1
            and _expandable_segments_enabled()
        ):
            vllm_args["disable_custom_all_reduce"] = True
        
        if enable_lora:
            if int(self.config.lora_r) not in _SUPPORTED_VLLM_LORA_RANKS:
                raise ValueError(
                    f"vLLM {getattr(vllm, '__version__', 'unknown')} only supports "
                    f"LoRA ranks {_SUPPORTED_VLLM_LORA_RANKS} in this stack, got {self.config.lora_r}."
                )
            vllm_args["enable_lora"] = True
            vllm_args["max_lora_rank"] = self.config.lora_r
        else:
            vllm_args["enable_lora"] = False

        # Set gpu_memory_utilization from current free memory while retaining headroom.
        util = float(self.config.gpu_memory_utilization)
        util_cap = 0.90
        
        if is_single_gpu_job:
            util_cap = 0.65 if (not self.eval_mode and self.num_samples > 1) else 0.78
        if (
            is_checkpoint_rollout
        ):
            util_cap = min(util_cap, min(0.25, max(0.23, util)))
            print(
                "[Actor] Keeping gpu_memory_utilization "
                f"at {util_cap:.2f} for checkpoint rollouts"
            )
        
        if torch.cuda.is_available():
            try:
                tp_size = max(1, int(self.config.tensor_parallel_size))
                headroom = int(1.0 * 1024**3)
                max_utils = []
                for device_idx in range(tp_size):
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
                    max_utils.append((free_bytes - headroom) / float(total_bytes))
                max_util = min(max_utils)
                max_util = max(0.05, min(max_util, util_cap))
                if abs(max_util - util) > 0.02:
                    print(f"[Actor] Adjusting gpu_memory_utilization {util:.2f} -> {max_util:.2f} based on free memory")
                    util = max_util
            except RuntimeError as err:
                print(f"[Actor] Could not check free GPU memory; using configured utilization: {err}")
        
        util_candidates = [util] + [u for u in (0.23, 0.2, 0.18, 0.15) if u < util]
        last_err = None
        
        for u in util_candidates:
            vllm_args["gpu_memory_utilization"] = u
            try:
                self.llm = vllm.LLM(**vllm_args)
                if u != util:
                    print(f"[Actor] vLLM retry succeeded with gpu_memory_utilization={u}")
                    self.config.gpu_memory_utilization = u
                break
            except Exception as err:
                last_err = err
                msg = str(err)
                if (
                    "Free memory on device" not in msg
                    and "gpu memory utilization" not in msg
                    and "Engine core initialization failed" not in msg
                ):
                    raise
        
        if self.llm is None and last_err is not None:
            raise last_err

        # IMPORTANT: use vLLM tokenizer for generation formatting
        self.tokenizer = self.llm.get_tokenizer()
        self._ensure_padding()
        self.sampling_params = self._build_vllm_sampling_params()
        self.vllm_lora_enabled = enable_lora

    def _build_vllm_sampling_params(self):
        assert self.tokenizer is not None
        # Get stop token IDs for this model (important for Llama Instruct)
        stop_token_ids = _get_stop_token_ids(self.tokenizer, self.config.pretrain_model)

        sampling_kwargs = {
            "max_tokens": self.max_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k if self.config.top_k > 0 else -1,
            "temperature": self.temperature,
            "n": self.num_samples,
        }
        if stop_token_ids:
            sampling_kwargs["stop_token_ids"] = stop_token_ids

        return vllm.SamplingParams(**sampling_kwargs)

    def generate(self, prompts: List[str]) -> Dict[int, List[str]]:
        return self._generate_vllm(prompts)

    def _generate_vllm(self, prompts: List[str]) -> Dict[int, List[str]]:
        assert self.llm is not None
        assert self.tokenizer is not None
        assert self.sampling_params is not None

        lora_req = None
        cur_path = self.model_path_override or self.config.current_model_path
        if self.config.use_lora and _is_lora_adapter_dir(cur_path) and not self.vllm_lora_enabled:
            raise RuntimeError(
                "Persistent vLLM engine was initialized without LoRA support but "
                f"current_model_path is now an adapter directory: {cur_path}"
            )
        if self.config.use_lora and _is_lora_adapter_dir(cur_path) and LoRARequest is not None:
            adapter_name = os.path.basename(cur_path.rstrip("/"))
            adapter_id = zlib.crc32(cur_path.encode("utf-8")) & 0x7FFFFFFF
            lora_req = LoRARequest(adapter_name, adapter_id, cur_path)

        outputs = self.llm.generate(
            prompts,
            self.sampling_params,
            use_tqdm=False,
            lora_request=lora_req,
        )

        candidates: Dict[int, List[str]] = {}
        n = self.sampling_params.n
        for i, out in enumerate(outputs):
            # vLLM returns ONLY generated continuation in out.outputs[j].text
            candidates[i] = [
                out.outputs[j].text.strip()
                for j in range(n)
            ]
        return candidates

    def cleanup(self):
        """Release the vLLM engine and associated GPU memory."""
        if self.llm is not None:
            # Preferred: vLLM LLM shutdown (releases engine workers + GPU mem)
            try:
                if hasattr(self.llm, "shutdown"):
                    self.llm.shutdown()
            except Exception as err:
                print(f"[Actor] vLLM shutdown failed: {err}")

            self.llm = None
            self.vllm_lora_enabled = False

        # Python/CUDA cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
