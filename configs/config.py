from dataclasses import dataclass, field
import os
import torch
from typing import Literal, Optional

# Dataset type detection
DatasetType = Literal["tldr", "ultrafeedback", "unknown"]

def detect_dataset_type(prompt_data: str) -> DatasetType:
    """Detect dataset type from dataset name."""
    lower = prompt_data.lower()
    if "tldr" in lower:
        return "tldr"
    elif "ultrafeedback" in lower:
        return "ultrafeedback"
    return "unknown"

def is_llama_model(model_name: str) -> bool:
    """Check if model is a Llama variant."""
    lower = model_name.lower()
    return "llama" in lower or "meta-llama" in lower

def is_llama_1b_model(model_name: str) -> bool:
    """Check if model is a 1B-class Llama variant."""
    lower = model_name.lower()
    return is_llama_model(model_name) and "1b" in lower

def is_instruct_model(model_name: str) -> bool:
    """Check if model is instruction-tuned (requires chat template)."""
    lower = model_name.lower()
    return any(x in lower for x in ["instruct", "chat", "it", "zephyr"])

@dataclass
class AeroConfig:

    pretrain_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    prompt_data: str = "HuggingFaceH4/ultrafeedback_binarized"
    prompt_data_dir: Optional[str] = None
    oracle: str = "Skywork/Skywork-Reward-Llama-3.1-8B"

    prompt_max_length: int = 1024
    generate_max_length: int = None  # Auto-detect based on dataset

    apply_chat_template: bool = None  # None = auto-detect

    #vllm rollout
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    num_samples: int = 20

    #exploration
    burn_in_period: int = 8
    max_model_data_ratio: float = 0.30
    max_iterations: int = 500
    seed: int = 42  # random seed for reproducibility
    use_aero_explorer: bool = True
    use_online_dpo: bool = False  # Vanilla online DPO baseline
    use_copo: bool = False
    copo_mode: Literal["copo_fixed", "copo_rolling"] = "copo_fixed"
    use_aero_variant2: bool = False  # AERO variant 2: fixed-ratio random oracle/ERM routing
    use_aero_variant3: bool = False  # AERO variant 3: SEA-style synthetic branch selection and labeling
    use_aero_variant4: bool = False  # AERO variant 4: confidence-gated synthetic labeling
    use_aero_fixed_quantile: bool = False  # Fixed top-q vote-entropy routing
    aero_variant2_oracle_ratio: float = 0.44  # Paper Variant 2 allocation before filtering, after burn-in.
    aero_fixed_quantile_ratio: float = 0.46

    # CoPO hyperparameters (only used if use_copo=True)
    copo_alpha: float = 0.001
    copo_stage1_num_prompts: int = 5000
    copo_num_samples: int = 3
    copo_burnin_num_samples: int = 2
    copo_cfn_batch_size: int = 2
    copo_cfn_lr: float = 1e-4
    copo_bonus_exponent: float = 0.5
    copo_cfn_buffer_maxlen: int = 5000

    # AERO-specific
    confidence_filter: bool = False
    confidence_threshold:float = 0.85
    # Counterfactual audit: query the oracle on ERM-routed pairs without using
    # those labels for ERM/DPO training or the reported training-query budget.
    # used in the 100 iteration audit run during rebuttal
    audit_synthetic_labels: bool = False

    # Aleatoric filtering: exclude high-entropy (aleatoric) samples from ERM training
    aleatoric_filter: bool = True
    aleatoric_threshold: float = 0.6  # mean_H > threshold → aleatoric

    # DPO
    algo_beta: float = 0.1
    lr: float = 5e-7
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.95
    l2: float = 0.0
    max_grad_norm: float = 1.0
    dpo_epochs_per_iter: int = 1

    lr_scheduler: str = "cosine_with_min_lr"
    lr_warmup_ratio: float = 0.02
    min_lr_ratio: float = 0.1  # min_lr = lr * 0.1
    max_step_adjustment: float = 0.75

    train_batch_size: int = 128
    train_batch_size_per_device: int = 2
    pi_buffer_maxlen_per_device: int = 128
    dpo_effective_batch_size: Optional[int] = None

    # ERM
    encoder: str = "llm-blender/PairRM-hf"
    encoding_dim: int = 2048
    num_ensemble: int = 20
    enn_lambda: float = 0.5
    rm_hidden_dim: int = 128
    rm_act_fn: str = "relu"
    rm_sgd_steps: int = 5
    rm_train_bs: int = 128
    rm_lr: float = 1e-3
    rm_wd: float = 5e-5
    backbone_batch: int = 7
    max_length: int = 2048

    # vLLM
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.20
    trust_remote_code: bool = True

    use_lora: bool = False

    # LoRA hyperparameters.
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_bias: str = "none"
    use_flash_attention_2: bool = False

    # Swap oracle/ref models between GPUs during DPO to reduce GPU0 OOM risk
    swap_oracle_ref_during_dpo: bool = True
    
    # Dataset splits (auto-selected based on dataset type if None)
    train_split: str = None
    eval_split: str = None

    # Evaluation
    max_eval: int = 1000
    eval_batch_size: int = 8
    oracle_batch_size: int = 16
    eval_every_n_iters: int = 25
    eval_temperature: float = 0.0
    eval_generate_max_length: int = 512
    eval_reference_key: str = None  #auto-select based on model

    # Output paths
    output_dir: str = "./outputs"
    save_dir: str = "./outputs/checkpoints"
    current_model_path: str = field(init=False)

    # devices
    device_actor: str = field(init=False)
    device_oracle: str = field(init=False)
    device_ref: str = field(init=False)

    max_seq_len: int = field(init=False)
    grad_accum_steps: int = field(init=False)

    # Actor expects these:
    max_tokens: int = field(init=False)        # == generate_max_length
    eval_max_tokens: int = field(init=False)   # == eval_generate_max_length
    source_max_length: int = field(init=False) # == prompt_max_length

    train_batch: int = field(init=False)       # == train_batch_size
    r_buffer_maxlen: int = 50000               

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        if self.use_aero_variant4:
            self.confidence_filter = True

        if self.copo_mode not in {"copo_fixed", "copo_rolling"}:
            raise ValueError(f"Unsupported copo_mode: {self.copo_mode}")
        if not 0.0 <= float(self.aero_variant2_oracle_ratio) <= 1.0:
            raise ValueError(
                f"aero_variant2_oracle_ratio must be in [0, 1], got {self.aero_variant2_oracle_ratio}"
            )
        if not 0.0 <= float(self.aero_fixed_quantile_ratio) <= 1.0:
            raise ValueError(
                "aero_fixed_quantile_ratio must be in [0, 1], "
                f"got {self.aero_fixed_quantile_ratio}"
            )

        self.current_model_path = self.pretrain_model

        # --- Dataset Type Detection ---
        self._dataset_type = detect_dataset_type(self.prompt_data)
        print(f"[Config] Detected dataset type: {self._dataset_type}")
        if self.prompt_data_dir:
            print(f"[Config] Using dataset data_dir: {self.prompt_data_dir}")

        # --- Model Type Detection ---
        self._is_llama_1b = is_llama_1b_model(self.pretrain_model)
        self._is_instruct = is_instruct_model(self.pretrain_model)

        if (
            self._dataset_type == "ultrafeedback"
            and not self._is_llama_1b
            and self.prompt_max_length == 1024
        ):
            self.prompt_max_length = 512
            print("[Config] UltraFeedback default: prompt_max_length -> 512")

        # Auto-set generate_max_length based on dataset
        if self.generate_max_length is None:
            if self._dataset_type == "tldr":
                self.generate_max_length = 53  # TL;DR summaries are short
            elif self._dataset_type == "ultrafeedback":
                self.generate_max_length = 256 if self._is_llama_1b else 512
            else:
                self.generate_max_length = 128  # Default fallback
            print(f"[Config] Auto-set generate_max_length={self.generate_max_length} for {self._dataset_type}")        

        # Auto-set dataset splits based on dataset type
        if self.train_split is None:
            if self._dataset_type == "ultrafeedback":
                self.train_split = "train_prefs"  # UltraFeedback uses train_prefs for preference data
            else:
                self.train_split = "train"
            print(f"[Config] Auto-set train_split='{self.train_split}'")

        if self.eval_split is None:
            if self._dataset_type == "ultrafeedback":
                self.eval_split = "test_prefs"  # UltraFeedback uses test_prefs for preference eval
            else:
                self.eval_split = "test"
            print(f"[Config] Auto-set eval_split='{self.eval_split}'")

        self.max_seq_len = int(self.prompt_max_length + self.generate_max_length)

        # Auto-set apply_chat_template for instruct models
        if self.apply_chat_template is None:
            self.apply_chat_template = self._is_instruct
            if self._is_instruct:
                print(f"[Config] Detected instruct model, enabling chat template")

        # Auto-select eval reference key based on model and dataset
        if self.eval_reference_key is None:
            if self._dataset_type == "ultrafeedback":
                self.eval_reference_key = "sft-baseline"
            elif "pythia-1b" in self.pretrain_model.lower():
                self.eval_reference_key = "pythia-1b-reference"
            else:
                # For non-Pythia models (like Llama), use generated SFT baseline
                self.eval_reference_key = "sft-baseline"
            print(f"[Config] Auto-selected eval_reference_key='{self.eval_reference_key}'")

        # --- Device Detection ---
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if n_gpus >= 2:
                self.device_actor = "cuda:0"
                self.device_oracle = "cuda:1"
                self.device_ref = "cuda:0"
                if self.use_lora:
                    # 2-GPU LoRA layout used for the 8B runs:
                    # cuda:0: policy + optimizer + ref
                    # cuda:1: oracle + ERM
                    self.device_erm = "cuda:1"
                    if self.train_batch_size_per_device <= 2:
                        self.train_batch_size_per_device = 4
                        print("[Config] LoRA 2-GPU: train_batch_size_per_device -> 4")
                    if self.backbone_batch <= 7:
                        self.backbone_batch = 16
                        print("[Config] LoRA 2-GPU: backbone_batch -> 16")
                    if self.oracle_batch_size <= 16:
                        self.oracle_batch_size = 32
                        print("[Config] LoRA 2-GPU: oracle_batch_size -> 32")
                    print("[Config] 2 GPUs with LoRA detected: dpo+ref=cuda:0, oracle+erm=cuda:1")
                else:
                    # 2-GPU non-LoRA layout used for the 1B runs:
                    # cuda:0: policy + ref + ERM
                    # cuda:1: oracle
                    self.device_erm = "cuda:0"
                    print("[Config] 2 GPUs detected: dpo+ref+erm=cuda:0, oracle=cuda:1")
            else:
                self.device_actor = "cuda:0"
                self.device_oracle = "cuda:0"
                self.device_ref = "cuda:0"
                self.device_erm = "cuda:0"
        elif torch.backends.mps.is_available():
            self.device_actor = "mps"
            self.device_oracle = "mps"
            self.device_ref = "mps"
            self.device_erm = "mps"
            n_gpus = 1
        else:
            self.device_actor = "cpu"
            self.device_oracle = "cpu"
            self.device_ref = "cpu"
            self.device_erm = "cpu"
            n_gpus = 1

        target_global_batch = (
            int(self.dpo_effective_batch_size)
            if self.dpo_effective_batch_size is not None
            else self.train_batch_size
        )

        training_gpus = 1
        total_micro_batch = self.train_batch_size_per_device * training_gpus

        if target_global_batch % total_micro_batch != 0:
            print(f"[Config] Warning: Global batch {target_global_batch} not divisible by "
                  f"micro batch {total_micro_batch}. Rounding down.")

        self.grad_accum_steps = max(1, target_global_batch // total_micro_batch)

        print(f"[Config] DPO Training: Single-GPU on {self.device_actor}")
        print(f"[Config] Effective Batch Size: {self.train_batch_size_per_device * training_gpus * self.grad_accum_steps}")
        print(f"[Config] GPUs detected: {n_gpus}, DPO Micro-Batch: {self.train_batch_size_per_device}, "
              f"Grad Accum: {self.grad_accum_steps}")

        self.max_tokens = int(self.generate_max_length)
        self.eval_max_tokens = int(self.eval_generate_max_length)
        self.source_max_length = int(self.prompt_max_length)
        self.train_batch = int(self.train_batch_size)
