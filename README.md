# AERO

Official implementation of **AERO: Adaptive Ensemble-Disagreement Routing for
Oracle Feedback in Sample-Efficient Online RLHF** (NeurIPS 2026).

AERO uses ensemble disagreement to route uncertain contexts to an oracle and
lower-disagreement contexts to an Epistemic Reward Model for synthetic
preference labels.

## Installation

Python 3.10-3.13 and a CUDA-compatible Linux system are required. Hugging Face
access to the gated Llama models is also required.

For CUDA 13 Blackwell systems such as the RTX PRO 6000:

```bash
bash scripts/setup_aws_env.sh
source venv_aws/bin/activate
hf auth login
```

The setup script creates a virtual environment, installs the project and its
CUDA dependencies, and checks that PyTorch and vLLM can use the GPU.

To install in an existing compatible environment:

```bash
python -m pip install -e .
hf auth login
```

## Running Experiments

The launcher contains the settings used for the paper. For example, run AERO
with Llama-3.2-1B and seed 42 using:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash recipes/launch_train.sh llama1b aero 42
```

Replace `aero` with `sea`, `copo`, or `online_dpo` to run the main baselines.
The available model profiles are `llama1b`, `qwen15b`, `llama8b`, and
`pythia_tldr`.

Running `python main.py` without arguments starts AERO with seed 42 on
Llama-3.2-1B and UltraFeedback. The installed `aero-train` command provides the
same interface.

See [EXPERIMENTS.md](EXPERIMENTS.md) for all paper experiments, ablations,
seeds, and cluster examples.

## Repository Structure

- `configs/`: training and model configuration.
- `recipes/`: experiment launchers for local and Slurm runs.
- `scripts/exploration/`: AERO and baseline exploration methods.
- `scripts/reward/`: reward-model backbone and ensemble.
- `scripts/preference/`: preference learning and policy optimization.
- `scripts/trainer.py`: online training loop and diagnostics.
- `main.py`: command-line entry point.

## Acknowledgement

`scripts/reward/ensemble.py` and portions of `scripts/reward/reward_model.py`
and `scripts/reward/RMbackbone.py` are copied or adapted from
[OAT](https://github.com/sail-sg/oat) under the Apache License 2.0.
These files retain the original license headers; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

## Citation

If you use this code, please cite AERO:

```bibtex
@inproceedings{singhal2026aero,
  title     = {AERO: Adaptive Ensemble-Disagreement Routing for Oracle Feedback in Sample-Efficient Online RLHF},
  author    = {Singhal, Shrey and Huang, Nannan and Fayek, Haytham M.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

For the reused OAT code, please also cite:

```bibtex
@misc{liu2024oat,
  title        = {OAT: A Research-Friendly Framework for LLM Online Alignment},
  author       = {Liu, Zichen and Chen, Changyu and Wan, Xinyi and Du, Chao and Lee, Wee Sun and Lin, Min},
  year         = {2024},
  howpublished = {\url{https://github.com/sail-sg/oat}}
}
```
