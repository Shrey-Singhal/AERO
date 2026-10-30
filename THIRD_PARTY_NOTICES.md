# Third-Party Notices

This repository incorporates source code from third-party open-source projects.
The relevant copyright notices and license terms are reproduced below, in
accordance with the terms of the respective licenses.

## Oat — Online LLM Alignment Framework

- **Project:** Oat (`sail-sg/oat`)
- **Source:** https://github.com/sail-sg/oat
- **Copyright:** Copyright 2024 Garena Online Private Limited
- **License:** Apache License, Version 2.0 (full text: [`licenses/oat-Apache-2.0.txt`](licenses/oat-Apache-2.0.txt))

Portions of the reward-modeling code in this repository are copied or adapted
from Oat. Each such file retains the original Apache-2.0 header and carries a
notice of modification. The affected files are:

| File in this repository | Corresponding Oat source | Relationship |
| --- | --- | --- |
| `scripts/reward/ensemble.py` | `oat/oat/rm/networks.py` | Copied (ensemble ENN networks) |
| `scripts/reward/RMbackbone.py` | `oat/oat/rm/backbone.py` | Adapted (PairRM feature backbone) |
| `scripts/reward/reward_model.py` | `oat/oat/rm/model.py` | Adapted (ensemble reward model / ENN learning) |

Additionally, the following files implement methods from, or reuse small
non-substantial idioms of, Oat, and credit it inline rather than carrying the
full license header:

- `scripts/exploration/SEAExplorer.py` — an independent re-implementation of the
  **SEA / EnnBAITS** exploration method used as a baseline. The method is
  introduced in *Sample-Efficient Alignment for LLMs* (Liu et al., 2024;
  arXiv:2411.01493), cited in the file's module docstring.
- `scripts/preference/preference_learning.py` — adapts the circular-buffer
  sampling scheme from `oat/oat/utils/buffer.py`.

See the "Acknowledgements" and "Citation" sections of the [README](README.md)
for the full references.
