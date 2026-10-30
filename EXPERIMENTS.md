# Paper Experiments

Run these commands from the repository root, in your activated training
environment. The launcher selects the settings explicitly; you do not need to
edit `configs/config.py`. Bare `python main.py` defaults to original AERO with
seed `42` on Llama-3.2-1B / UltraFeedback.

For AWS environment installation, follow the setup steps in
[README.md](README.md#prerequisites). Setup and training are separate commands.

```bash
bash recipes/launch_train.sh PROFILE EXPERIMENT SEED
```

## Training Settings

| Profile | Model / data | Iterations | Policy tuning |
| --- | --- | ---: | --- |
| `llama1b` | Llama-3.2-1B-Instruct / UltraFeedback | 500 | Full, LR `5e-7` |
| `qwen15b` | Qwen2.5-1.5B-Instruct / UltraFeedback | 500 | Full, LR `5e-7` |
| `llama8b` | Meta-Llama-3-8B-Instruct / UltraFeedback | 300 | LoRA, LR `2e-5` |
| `pythia_tldr` | Pythia-1B TL;DR SFT / TL;DR | 500 | Full, LR `5e-7` |

Prompt length and training/evaluation generation limits come from `configs/config.py`;
the launcher does not override them for any model.

Shared settings are batch size 128, eight burn-in iterations, training
temperature `0.7`, top-p `1.0`, DPO beta `0.1`, and one DPO epoch per
iteration. AERO and SEA use 20 candidates and 20 reward heads. COPO uses three
candidates and a fixed CFN by default; Online DPO uses two candidates.

The oracle is `Skywork/Skywork-Reward-Llama-3.1-8B` and the ERM encoder is
`llm-blender/PairRM-hf`. LoRA uses rank 64, alpha 64, and dropout `0.05`.
Learning-rate warmup is `0.02`, or `0.005` for the 8B LoRA profile.

Internal evaluation uses greedy decoding and up to 1,000 held-out rows before
prompt-length filtering, every 25 iterations.
UltraFeedback evaluation compares against cached initial-policy responses;
TL;DR uses the dataset's `pythia-1b-reference`.

The paper used two A100 40GB GPUs per Llama-1B run, one RTX PRO 6000 96GB GPU
per Qwen run, and two RTX PRO 6000 96GB GPUs per Llama-8B run. Select available
GPUs with `CUDA_VISIBLE_DEVICES`; under Slurm, use the scheduler's allocation.

## Main Experiments

Figures 2 and 4, and the training checkpoints used in Table 1:

```bash
for seed in 0 1 42; do
  for method in aero sea copo online_dpo; do
    CUDA_VISIBLE_DEVICES=0,1 bash recipes/launch_train.sh llama1b "$method" "$seed"
    CUDA_VISIBLE_DEVICES=0 bash recipes/launch_train.sh qwen15b "$method" "$seed"
  done
done

for method in aero sea copo online_dpo; do
  CUDA_VISIBLE_DEVICES=0,1 bash recipes/launch_train.sh llama8b "$method" 42
done
```

Loops run sequentially. Each invocation starts one experiment.

## Component Ablations

Figure 3 uses Llama-1B / UltraFeedback and seeds 0, 1, and 42. Reuse the main
AERO runs for the reference curve.

| Experiment | Change from AERO |
| --- | --- |
| `variant1` | Disable aleatoric pair filtering |
| `variant2` | Random oracle/ERM routing at allocation `0.44` |
| `variant3` | SEA-style single-head synthetic selection and labeling |
| `variant4` | Route synthetic pairs below confidence `0.85` back to the oracle |

```bash
for seed in 0 1 42; do
  for variant in variant1 variant2 variant3 variant4; do
    CUDA_VISIBLE_DEVICES=0,1 bash recipes/launch_train.sh llama1b "$variant" "$seed"
  done
done
```

## Fixed-Quantile Routing

Figure 7 retains vote-entropy ranking and replaces Otsu with fixed allocation
`0.46`. This is different from Variant 2's random routing at `0.44`.
Both allocations apply after burn-in and before pair filtering.

```bash
for seed in 0 1 42; do
  ORACLE_RATIO=0.46 CUDA_VISIBLE_DEVICES=0,1 bash recipes/launch_train.sh llama1b fixed_quantile "$seed"
done

# Table 5: the paper's single-run allocation sweep.
for ratio in 0.15 0.25 0.65; do
  ORACLE_RATIO="$ratio" CUDA_VISIBLE_DEVICES=0,1 bash recipes/launch_train.sh llama1b fixed_quantile 42
done
```

Table 5 reports iterations 100, 200, 300, 400, and 500, alongside the
minimum-maximum across the original three AERO runs. The 27K/30K matched-budget
ranges are analysis ranges, not training stopping conditions.

## Synthetic-Label and Entropy Audit

Appendix C.2 uses a separate 100-iteration AERO run:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash recipes/launch_train.sh llama1b audit 42
```

Every retained synthetic pair is queried from the oracle for measurement only.
Audit labels never enter policy/ERM training or the training-query budget.
This recipe evaluates up to 128 raw held-out rows at the final iteration.

The audit retains all context-level vote entropies. Excluding eight burn-in
iterations leaves 92 batches for the entropy/dip-test analysis. Overall
synthetic agreement is total matching labels divided by total audited pairs.

Figures 5 and the other Appendix C.1/C.2 diagnostics use the normal AERO runs:
routing counts, filtering, uncertainty, head agreement, and synthetic confidence
are already logged. They do not require separate training.

## COPO Fixed Versus Rolling

Figure 8 reuses the main fixed-CFN COPO runs and adds rolling-CFN runs:

```bash
for seed in 1 42; do
  CUDA_VISIBLE_DEVICES=0,1 bash recipes/launch_train.sh llama1b copo_rolling "$seed"
done
```

## Pythia / TL;DR

Figure 9 uses single-run comparisons:

```bash
for method in aero sea copo online_dpo; do
  CUDA_VISIBLE_DEVICES=0,1 bash recipes/launch_train.sh pythia_tldr "$method" 42
done
```

The profile uses `trl-lib/pythia-1b-deduped-tldr-sft` and
`lkevinzc/tldr-with-sft-reference`, recovered from the historical config.
The archived SEA trace used an older budget-based stopping rule and evaluation
cadence and was cropped to 500 iterations for plotting. This recipe uses a
common 500-iteration schedule, not an exact replay of that older stopping rule.

## AWS and Slurm

The training recipe is the same on both systems. For an AWS background run:

```bash
mkdir -p outputs/logs
CUDA_VISIBLE_DEVICES=0 nohup bash recipes/launch_train.sh qwen15b aero 42 \
  > outputs/logs/qwen15b_aero_seed42.log 2>&1 &
```

For Slurm, activate your environment before submitting from the repository root:

```bash
sbatch recipes/slurm_train.sh llama1b aero 42
```

The script requests two GPUs, 64GB RAM, and 36 hours. Supply your cluster's
partition/account options before the script path. Do not overwrite Slurm's
`CUDA_VISIBLE_DEVICES`. Set `PYTHON_BIN` to an absolute venv Python path if needed.

Additional `main.py` options can be appended to either launcher, for example
`--output-dir outputs/my_run`. Use a fresh output directory for repeated runs
so checkpoints and cached references are not reused or overwritten.
