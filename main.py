import argparse
import os

from configs.config import AeroConfig
from scripts.trainer import AEROTrainer


def _normalize_method(method: str) -> str:
    normalized = method.lower().strip()
    aliases = {
        "aero": "aero",
        "sea": "sea",
        "copo": "copo",
        "dpo": "online_dpo",
        "online_dpo": "online_dpo",
        "aero_variant2": "aero_variant2",
        "aero_variant3": "aero_variant3",
        "aero_variant4": "aero_variant4",
        "aero_fixed_quantile": "aero_fixed_quantile",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown method '{method}'. Use aero, sea, copo, dpo, "
            "aero_variant2, aero_variant3, aero_variant4, or aero_fixed_quantile."
        )
    return aliases[normalized]


def _apply_method(cfg: AeroConfig, method: str) -> str:
    normalized = _normalize_method(method)
    cfg.use_online_dpo = False
    cfg.use_copo = False
    cfg.use_aero_explorer = False
    cfg.use_aero_variant2 = False
    cfg.use_aero_variant3 = False
    cfg.use_aero_variant4 = False
    cfg.use_aero_fixed_quantile = False
    cfg.confidence_filter = False

    if normalized == "online_dpo":
        cfg.use_online_dpo = True
    elif normalized == "copo":
        cfg.use_copo = True
    elif normalized == "aero":
        cfg.use_aero_explorer = True
    elif normalized == "aero_variant2":
        cfg.use_aero_variant2 = True
    elif normalized == "aero_variant3":
        cfg.use_aero_variant3 = True
    elif normalized == "aero_variant4":
        cfg.use_aero_variant4 = True
        cfg.confidence_filter = True
    elif normalized == "aero_fixed_quantile":
        cfg.use_aero_fixed_quantile = True

    return normalized


def _current_method(cfg: AeroConfig) -> str:
    if cfg.use_online_dpo:
        return "online_dpo"
    if cfg.use_copo:
        return "copo"
    if cfg.use_aero_fixed_quantile:
        return "aero_fixed_quantile"
    if cfg.use_aero_variant2:
        return "aero_variant2"
    if cfg.use_aero_variant4:
        return "aero_variant4"
    if cfg.use_aero_variant3:
        return "aero_variant3"
    if cfg.use_aero_explorer:
        return "aero"
    return "sea"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configurable AERO-RLHF experiment.")
    parser.add_argument(
        "--method",
        choices=[
            "aero",
            "sea",
            "copo",
            "dpo",
            "online_dpo",
            "aero_variant2",
            "aero_variant3",
            "aero_variant4",
            "aero_fixed_quantile",
        ],
        default=None,
        help="Explorer/method to run.",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt-max-length", type=int, default=None)
    parser.add_argument("--generate-max-length", type=int, default=None)
    parser.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the LoRA policy path. Must be set at config construction time so GPU layout adjusts correctly.",
    )
    parser.add_argument(
        "--use-flash-attention-2",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Request FlashAttention-2 for HuggingFace policy loads when supported.",
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--eval-every-n-iters", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-warmup-ratio", type=float, default=None)
    parser.add_argument("--max-model-data-ratio", type=float, default=None)
    parser.add_argument("--copo-mode", choices=["copo_fixed", "copo_rolling"], default=None)
    parser.add_argument("--copo-alpha", type=float, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--eval-generate-max-length", type=int, default=None)
    parser.add_argument("--eval-reference-key", type=str, default=None)
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--eval-split", type=str, default=None)
    parser.add_argument(
        "--aleatoric-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable pair filtering; --no-aleatoric-filter is paper Variant 1.",
    )
    parser.add_argument("--aero-variant2-oracle-ratio", type=float, default=None)
    parser.add_argument("--aero-fixed-quantile-ratio", type=float, default=None)
    parser.add_argument(
        "--audit-synthetic-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Query the oracle on every AERO ERM-routed pair for diagnostics only. "
            "Audit labels never enter training or the training-query budget."
        ),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("AERO-RLHF Training")
    print("=" * 60)

    cfg_kwargs = {}
    if args.model is not None:
        cfg_kwargs["pretrain_model"] = args.model
    if args.dataset is not None:
        cfg_kwargs["prompt_data"] = args.dataset
    if args.dataset_dir is not None:
        cfg_kwargs["prompt_data_dir"] = args.dataset_dir
    if args.seed is not None:
        cfg_kwargs["seed"] = int(args.seed)
    if args.prompt_max_length is not None:
        cfg_kwargs["prompt_max_length"] = int(args.prompt_max_length)
    if args.generate_max_length is not None:
        cfg_kwargs["generate_max_length"] = int(args.generate_max_length)
    if args.use_lora is not None:
        cfg_kwargs["use_lora"] = bool(args.use_lora)
    if args.use_flash_attention_2 is not None:
        cfg_kwargs["use_flash_attention_2"] = bool(args.use_flash_attention_2)
    if args.aero_variant2_oracle_ratio is not None:
        cfg_kwargs["aero_variant2_oracle_ratio"] = float(args.aero_variant2_oracle_ratio)
    if args.aero_fixed_quantile_ratio is not None:
        cfg_kwargs["aero_fixed_quantile_ratio"] = float(args.aero_fixed_quantile_ratio)
    if args.audit_synthetic_labels is not None:
        cfg_kwargs["audit_synthetic_labels"] = bool(args.audit_synthetic_labels)
    for name in (
        "lr",
        "lr_warmup_ratio",
        "max_model_data_ratio",
        "copo_mode",
        "copo_alpha",
        "confidence_threshold",
        "eval_generate_max_length",
        "eval_reference_key",
        "train_split",
        "eval_split",
        "aleatoric_filter",
    ):
        value = getattr(args, name)
        if value is not None:
            cfg_kwargs[name] = value
    if args.output_dir is not None or args.run_name is not None:
        output_dir = args.output_dir or os.path.join("outputs", "runs", args.run_name)
        cfg_kwargs["output_dir"] = output_dir
        cfg_kwargs["save_dir"] = os.path.join(output_dir, "checkpoints")

    cfg = AeroConfig(**cfg_kwargs)
    normalized_method = _current_method(cfg)
    if args.method is not None:
        normalized_method = _apply_method(cfg, args.method)

    if args.max_iterations is not None:
        cfg.max_iterations = int(args.max_iterations)
    if args.eval_every_n_iters is not None:
        cfg.eval_every_n_iters = int(args.eval_every_n_iters)
    if args.max_eval is not None:
        cfg.max_eval = int(args.max_eval)
    if cfg.audit_synthetic_labels and normalized_method != "aero":
        raise ValueError("--audit-synthetic-labels is currently supported only with --method aero")

    print(f"Method: {normalized_method}")
    print(f"Seed: {cfg.seed}")
    print(f"Model: {cfg.pretrain_model}")
    print(f"Dataset: {cfg.prompt_data}")
    if cfg.prompt_data_dir:
        print(f"Dataset data_dir: {cfg.prompt_data_dir}")
    print(f"Output dir: {cfg.output_dir}")
    print(f"Visible CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    trainer = AEROTrainer(cfg)
    trainer.train()

    print(f"\n{'=' * 60}")
    print("Training completed successfully!")
    print(f"Results saved to: {cfg.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
