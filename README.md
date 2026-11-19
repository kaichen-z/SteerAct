<div align="center">

# SteerAct
### Unlocking Action Models through Few-Shot Temporal Steering

**One temporal descriptor connects feature selection and online steering.**

[![Paper](https://img.shields.io/badge/arXiv-2610.25632-b31b1b?style=for-the-badge)](https://arxiv.org/pdf/2610.25632)
[![Project Page](https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge)](https://steeract.github.io/)
[![Robot Demos](https://img.shields.io/badge/Robot-Demos-0f766e?style=for-the-badge)](https://steeract.github.io/#real-robots)
[![Fast-WAM Checkpoints](https://img.shields.io/badge/Hugging%20Face-Fast--WAM%20Checkpoints-ffcc33?style=for-the-badge)](https://huggingface.co/Yukiiii1234/ckpt)
[![Method](https://img.shields.io/badge/Explore-the_Method-475569?style=for-the-badge)](https://steeract.github.io/#method)

[Kaichen Zhou](https://kaichen-z.github.io/)<sup>1,2,&#42;</sup>, Zhiyan Li<sup>2,&#42;</sup>, Chongkai Gao<sup>3</sup>, Yuhan Wang<sup>4</sup>,<br>
Wenlong Huang<sup>3</sup>, Maggie Wang<sup>3</sup>, Fangneng Zhan<sup>1</sup>, Manling Li<sup>5</sup>,<br>
Qianqian Wang<sup>2</sup>, Mac Schwager<sup>3</sup>, Zhuang Liu<sup>6</sup>,<br>
Li Fei-Fei<sup>3</sup>, Mengyu Wang<sup>2,†</sup>, Paul Pu Liang<sup>1,†</sup>

<sup>1</sup> MIT &nbsp; <sup>2</sup> Harvard University &nbsp; <sup>3</sup> Stanford University<br>
<sup>4</sup> Imperial College London &nbsp; <sup>5</sup> Northwestern University &nbsp; <sup>6</sup> Princeton University<br>
<sup>&#42;</sup> Equal contribution &nbsp; <sup>†</sup> Equal supervision

[Overview](#overview) · [Quick Start](#quick-start) · [Configuration](#configuration) · [Workflow](#workflow) · [Release Scope](#release-scope) · [Citation](#citation)

</div>

<p align="center">
  <a href="https://steeract.github.io/#method">
    <img src="https://steeract.github.io/assets/images/steeract-method-v5.png" width="1000" alt="SteerAct: offline reference-bank construction and feature selection, followed by online support and failure checks and signed steering.">
  </a>
</p>

## Overview

**Can a few successful and failed trials tell a frozen robot action model what to change—and when?**

SteerAct uses a pretrained sparse autoencoder (SAE) to expose internal features and their steering directions. A shared temporal descriptor summarizes how each feature's activity rises, falls, or fluctuates, allowing comparisons across rollouts without aligning individual timesteps.

- **Before steering — select what to change.** Use successful and failed reference rollouts to rank features by outcome separation and construct task-specific reference banks.
- **During execution — decide when to intervene.** Compare the observed feature history with these banks. Intervene only when the history has sufficient reference support and resembles failure.
- **When the failure gate opens — reset and steer.** Restart the scene once with the same seed. Use the temporal activation history to match successful references and compute a bounded signed correction at each selected layer.

The paper studies single-feature steering using **24 outcome-labeled reference rollouts per task**, across VLA and WAM backbones. See the [project page](https://steeract.github.io/) for the paper's broader evaluation and robot videos.

> **Current code scope.** This folder provides collection, SAE training, feature ranking, and steering evaluation for **π₀.₅ and Fast-WAM on nine RoboTwin 2.0 tasks**. It requires external simulator/model integrations and checkpoints; it is not a self-contained reproduction of every paper experiment. See [Release Scope](#release-scope).

> **Single-feature setting.** The supplied recipes set `steering.top_k` to **1** per selected layer. `sae.top_k = 100` controls SAE sparsity.

## Quick Start

### 1. Prepare the environment

Use the **Linux/CUDA environment for your selected RoboTwin model integration**. Install its simulator assets, model dependencies, and compatible source checkouts first. The exact integration files required by this package are listed under [External integrations](#external-integrations).

The Python package requires **Python ≥ 3.10**, **NumPy ≥ 1.24**, **PyTorch ≥ 2.1**, and **huggingface_hub ≥ 0.20**. Once these are available in the model environment, install this package without changing that environment's dependency versions:

```bash
cd /path/to/SteerAct
python -m pip install -e . --no-deps
python -m steeract --help
```

`--no-deps` assumes the declared dependencies are already installed; it does not install the simulator or either action model. Collection and policy inference require CUDA as currently implemented. `--device cpu` can be used for offline SAE training/ranking, but does **not** move the robot policy to CPU.

### 2. Choose and edit a recipe

For a run covering all nine tasks, start with an `all_tasks` recipe:

```bash
STEERACT_CONFIG=configs/pi05_all_tasks.json
# For Fast-WAM instead:
# STEERACT_CONFIG=configs/fastwam_all_tasks.json
```

Before proceeding, edit the selected JSON file:

1. Set the local checkpoint and its matching normalization or statistics. Fast-WAM checkpoints are available on [Hugging Face](https://huggingface.co/Yukiiii1234/ckpt).
2. Set the RoboTwin checkout path, and the Fast-WAM checkout path if applicable.
3. Choose a fresh `paths.run_dir` for generated artifacts.
4. Choose [steering layers](#choose-steering-layers) and a [reference rollout budget](#choose-the-reference-rollout-budget).

All local paths are resolved **relative to the JSON file**, not the shell's working directory. Absolute paths are recommended. See [Configuration](#configuration) for required fields.

Inspect the resolved recipe:

```bash
python -m steeract collect --config "$STEERACT_CONFIG" --dry-run
```

**A dry run checks configuration structure only.** It does not verify that checkpoints, simulator assets, dependencies, or external adapters exist.

### 3. Run the four stages

Run these commands from the repository root in the prepared model environment:

```bash
# Collect outcome-labeled rollouts and hidden activations.
CUDA_VISIBLE_DEVICES=0 python -m steeract collect --config "$STEERACT_CONFIG"

# Train the SAE on the collected activations.
CUDA_VISIBLE_DEVICES=0 python -m steeract train --config "$STEERACT_CONFIG" --device cuda:0

# Select steering features and construct task-specific reference banks.
CUDA_VISIBLE_DEVICES=0 python -m steeract rank --config "$STEERACT_CONFIG" --device cuda:0

# Evaluate the frozen action model with online steering.
CUDA_VISIBLE_DEVICES=0 python -m steeract infer --config "$STEERACT_CONFIG" --device cuda:0
```

The same sequence supports all four recipes. Use a `single_task` recipe for one task. Each recipe trains a separate SAE for each selected task and layer.

## Configuration

### Available recipes

| Recipe | Backbone | SAE training data |
| :--- | :--- | :--- |
| [`pi05_all_tasks.json`](configs/pi05_all_tasks.json) | π₀.₅ | 9 task-specific sets of 24 reference rollouts |
| [`fastwam_all_tasks.json`](configs/fastwam_all_tasks.json) | Fast-WAM | 9 task-specific sets of 96 reference rollouts |
| [`pi05_single_task.json`](configs/pi05_single_task.json) | π₀.₅ | One task; 24 reference rollouts |
| [`fastwam_single_task.json`](configs/fastwam_single_task.json) | Fast-WAM | One task; 96 reference rollouts |

### Paths and model assets

| Configuration field | Required input |
| :--- | :--- |
| `paths.run_dir` | Output directory for this experiment |
| `paths.robotwin` | Compatible RoboTwin checkout, including the integration files below |
| `paths.checkpoint` | Task/model checkpoint compatible with the selected policy configuration |
| `paths.norm` — π₀.₅ | Normalization asset directory for that checkpoint/dataset |
| `paths.fastwam` — Fast-WAM | Compatible Fast-WAM source checkout |
| `paths.stats` — Fast-WAM | Dataset statistics JSON for that checkpoint |
| `paths.dit`, `paths.vae`, `paths.text_encoder`, `paths.tokenizer` — Fast-WAM | Local model assets or the supplied `hf://` references |

Download the Fast-WAM checkpoint that matches your task and set `paths.checkpoint` to the local file. Set `paths.stats` to the matching dataset statistics JSON. The recipe's `hf://` entries resolve the backbone assets listed above. [Fast-WAM checkpoints on Hugging Face](https://huggingface.co/Yukiiii1234/ckpt)

### Choose steering layers

Choose any nonempty subset of the supported visual layers. The Fast-WAM recipes steer at `fastwam_video_L27` and `fastwam_video_L28` by default and accept `fastwam_video_L26` through `fastwam_video_L29`. The number of entries in `steering.layers` is the number of layers used for SAE training, feature ranking, and steering.

To capture activations only at the steering sites, set `collection.layers` to the same list as `steering.layers`. Set the top-level `layer` to one of the selected layers. Use the same recipe for all four stages.

For example, merge these fields into a Fast-WAM recipe to use L27 and L28:

```json
{
  "layer": "fastwam_video_L28",
  "collection": {"layers": ["fastwam_video_L27", "fastwam_video_L28"]},
  "steering": {"layers": ["fastwam_video_L27", "fastwam_video_L28"]}
}
```

Keep the other fields in the selected recipe. With multiple layers, `collect` captures all selected layers in the same rollouts, `train` and `rank` produce artifacts for each task and layer, and `infer` installs all selected steering hooks in the same rollout.

### Choose the reference rollout budget

Set `collection.episodes_per_task` to 24, 96, or another integer that provides enough rollouts for the selected task. For example, set the following fields under `collection`:

```json
{
  "episodes_per_task": 96,
  "max_attempts": 1000
}
```

`collection.max_attempts` must be at least `episodes_per_task`. Every selected layer uses the same collected episodes. The collection manifest records this budget, and training, ranking, and inference use matching artifacts. The budget is per task. Set `evaluation.episodes_per_task` separately for held-out evaluation. Use a fresh `paths.run_dir` when changing the reference budget or layer selection.

<details>
<summary><strong>Registered RoboTwin tasks</strong></summary>

```text
put_bottles_dustbin
stack_blocks_three
blocks_ranking_size
place_dual_shoes
dump_bin_bigbin
open_microwave
turn_switch
place_mouse_pad
pick_diverse_bottles
```

</details>

## Workflow

| Stage | What it does | Main output |
| :--- | :--- | :--- |
| `collect` | Runs the frozen policy and records outcome-labeled activation histories | Activation shards and manifest |
| `train` | Fits a TopK SAE for each selected task and layer | SAE checkpoints |
| `rank` | Compares success/failure temporal descriptors, selects features, and builds references | Rankings and steering bank |
| `infer` | Applies gated, signed corrections during held-out robot rollouts | Success summaries and controller traces |

### Reference data and SAE training

Each reference rollout must contain at least **five decisions**, so both triple spacings can be represented. Feature ranking requires at least **two successful and two failed rollouts** per task. Inspect the outcome split after collection and use a new collection recipe and run directory if either outcome has fewer than two rollouts.

The SAE defaults are a dictionary of size **8 × hidden dimension**, **TopK sparsity k = 100** (at most 100 nonzero entries per token), **15,000 training steps**, batch size **1,024**, and learning rate **3 × 10⁻⁴**. The SAE is then frozen for feature selection and steering; the pretrained action model is not fine-tuned by any of these four stages.

### Temporal descriptors and online steering

Each feature history is summarized by an **18-dimensional descriptor**: for each of two spacings, six local order-pattern frequencies, normalized entropy, tie frequency, and an availability indicator. Reference-bank separation identifies candidate steering features.

**Failure-gated Reset.** The policy first runs with observation hooks. The selected layers share a Support Gate driven by the highest support mass. If any layer's Failure Gate opens, the evaluator stops before executing that decision's action and restarts once with the same seed. If the gate stays closed for the whole rollout, the evaluator uses that rollout's outcome.

**Temporal mean steering.** After reset, each selected layer targets the successful references' mean activation at the current decision. When temporal matching finds suitable neighbors, the controller blends the matched mean and the overall success mean equally. It divides the target gap by the estimated activation gain, clips the signed coefficient to `alpha_max`, and re-encodes the edited activation. Each layer uses its own top-ranked feature in the supplied recipes.

For `steering.top_k > 1`, the implementation averages selected features' intensity histories and decoder directions, then uses **one aggregate bank and one coefficient**. It does not run independent controllers for each feature.

<details>
<summary><strong>Default controller settings</strong></summary>

Set overrides under `steering.controller`; an empty object uses the defaults in [`steeract/method.py`](steeract/method.py).

| Setting | Default |
| :--- | :--- |
| `posterior_samples` | 500 Dirichlet draws |
| `support_mass` | 0.65 |
| `support_k` / `support_quantile` | 3 / 0.95; support radius is 3 × this reference-distance quantile |
| `outcome_k` | 3, capped by the smaller bank |
| `min_windows` | 3 observed triples at each spacing |
| `support_open` / `support_close` | 1 passing / 2 failing checks |
| `failure_open` / `failure_close` | 2 passing / 2 failing checks |
| `alpha_max` | 10, in either direction |
| `min_effective_gain` | 0.001 |

Support comparisons use raw descriptor distances. Outcome comparisons use fixed, class-balanced coordinate scales. During steering, the controller skips the edit when the layer lacks support, the reference target is unavailable, or the gain is below `min_effective_gain`.

</details>

### Evaluation and artifact consistency

- **Held-out seeds:** collection starts at seed 0. Evaluation starts at seed 5000 and uses 100 valid episodes per task and variant. The configured seed ranges must be disjoint.
- **Step limits:** `evaluation.max_steps_multiplier` defaults to **2**. Match evaluation horizons and budgets when comparing against unsteered policies or external baselines.
- **Recipe changes:** changing the backbone, tasks, selected layer, collection settings, policy settings, or source-asset paths changes the collection identity. Use a fresh run directory and regenerate compatible artifacts.
- **Steering changes:** after changing `steering.top_k` or controller settings, rerun `rank` before `infer`. Existing evaluation directories reject incompatible configurations. For new controller/evaluation settings at the same feature count, archive or rename the existing `results/top<k>/` directory before rerunning; keep the compatible SAE and bank in place. Changing `run_dir` alone does not move those inputs.
- **Collection errors:** invalid/too-short episodes and backend exceptions are logged and raised. `max_attempts` is not an automatic recovery mechanism for these errors.

## Outputs

Generated artifacts are written under `paths.run_dir`:

```text
<run_dir>/
├── activations/
│   ├── manifest.json            # Recipe identity and collected episodes
│   └── ...                      # Per-task, per-episode activation shards
├── sae/
│   ├── model.pt                 # Frozen SAE used by rank and infer
│   └── training.json
├── ranking/
│   └── <task>.json              # Feature scores and selection
├── steering_bank.json           # Task-specific references and controller settings
└── results/
    └── top<k>/                  # top1 for single-feature steering; supplied default: top4
        ├── configuration.json
        ├── episodes.jsonl
        ├── summary.json
        └── traces/
```

Success rates in `summary.json` are stored as **fractions in [0, 1]**; multiply by 100 for percentages. The macro-average weights each evaluated task equally.

## Release Scope

| Component | Status in this folder |
| :--- | :--- |
| Activation collection, TopK SAE training, feature ranking, online controller | Included in `steeract/` |
| π₀.₅ / Fast-WAM adapters for the nine registered RoboTwin tasks | Included; require compatible external integrations |
| All-task and single-task recipes with per-task, per-layer SAEs | Included |
| Simulator assets, task-policy checkpoints, normalization/statistics | External; not bundled |
| Collected activations, trained SAEs, and reference banks | Generated by the workflow; no pretrained artifacts bundled |
| DexJoCo and real-robot runners | Not included in this folder |
| Action-expert steering and DSRL/COAST/DrVLA baseline runners | Not exposed by the current four-stage CLI |
| Pinned external source revisions, environment lockfile, automated test suite, license file | Not included |

### External integrations

The backend adapters expect the following **specific source layout**. A generic RoboTwin installation is not sufficient unless it includes these integrations:

```text
<paths.robotwin>/
├── scripts/eval_policy_xpolicylab.py          # Both backbones
├── description/task_instruction/<task>.json  # π₀.₅ task instructions
└── XPolicyLab/                              # π₀.₅
    ├── policy/Pi_05/model.py
    ├── policy/Pi_05/openpi/src/
    └── utils/process_data.py

<paths.fastwam>/                             # Fast-WAM
├── experiments/robotwin/fastwam_policy/deploy_policy.py
└── configs/sim_robotwin.yaml
```

The π₀.₅ adapter expects 18 PaliGemma layers, 18 action-expert layers, and task-instruction JSON files containing `full_description`. The Fast-WAM adapter expects 30 video blocks, 30 action blocks, and the `model.mot._apply_expert_post_block_tensor` interception API. Exact compatible source revisions are not pinned in this folder; verify these interfaces before launching an experiment.

### Legacy scripts

Use **`python -m steeract`** for the workflow documented here. The separate `SAE/` directory contains legacy wrappers that depend on an external `SAE/sae_pipeline` checkout and use different defaults and artifact formats. Its checkpoints and banks should not be mixed with those produced by `steeract/`.

### Fast-WAM RoboTwin checkpoint map and 30-episode protocol

The released Hugging Face repository `Yukiiii1234/ckpt` contains five Fast-WAM
task checkpoints. The task-to-checkpoint mapping is saved locally in
`/mnt/benchmark/lizhiyan/steeract/hf/fastwam_robotwin/map.json`. To create one
independent-SAE recipe per task, with 24 reference episodes, a single selected
feature, and 30 held-out evaluation episodes, run:

```bash
python scripts/prepare_fastwam_experiments.py \
  --map /mnt/benchmark/lizhiyan/steeract/hf/fastwam_robotwin/map.json \
  --checkpoint-dir /mnt/benchmark/lizhiyan/steeract/hf/fastwam_robotwin \
  --robotwin /path/to/RoboTwin --fastwam /path/to/FastWAM \
  --stats /path/to/robotwin_fastwam_dataset_stats.json \
  --model-root /mnt/benchmark/lizhiyan/steeract/models \
  --output /mnt/benchmark/lizhiyan/steeract/configs30
```

Run `collect`, `train`, `rank`, and `infer` for each generated JSON. The
external RoboTwin/Fast-WAM checkouts and shared dataset statistics are required;
the checkpoint repository does not contain them.

The local research setup uses the L28 post-block site. A same-observation,
same-noise probe of FastWAM revision `7faa711` measured action changes of
0.02568 (L27), 0.01257 (L28), and exactly zero (L29); the repeated unedited
action change was zero. The final L29 output is computed after the final
video K/V cache entry and is not consumed by action inference in this revision.
`scripts/probe_fastwam.py` reproduces this check before collection.

For this machine, activate `scripts/research_env.sh`. Progress is available with
`python scripts/research_status.py`; see [PROGRESS.md](PROGRESS.md) for the
protocol and external integration patches. `scripts/run_research.py --gpus 0 1`
resumes the prepared pipelines using independent SAEs and a single selected
feature. It starts with 24 reference episodes, extending to 48 or 96 only if
there are fewer than two successes or failures. A scene rejected as unstable
before the policy starts is logged separately; policy failures remain in the
denominator. The final evaluation uses 30 valid episodes from seeds 5000 onward.
The driver does not tune on final evaluation results. Summaries report each
task's absolute gap from the paper; all nine tasks must be present to pass.

After the first round, its original results remain frozen. Independent
development uses seeds 7000–7999; a new final evaluation must use untouched
seeds after selecting and freezing the method. Inspect current development
counts and actual live workers with `python scripts/research_status.py --development`.
`python scripts/compare_development.py` compares common valid seeds and reports
rescued/harmed pairs and the exact paired test; development scores do not count
as the final 30-trial result.

The optional controller setting `response_model: "measured"` searches the real
SAE response to each dtype-quantized candidate edit. `reference_mode: "prefix"`
uses banks at matching decision times, permits available early temporal windows,
and stops intervention outside their coverage. These banks must be built from
uncensored reference prefixes with `scripts/prepare_prefix_development.py`.
`scripts/screen_prefix_reachability.py` additionally checks whether the single
selected direction can actually reduce reference descriptor distance. These are
experimental alternatives; reference classification or reachability alone does
not establish closed-loop improvement.

The separate continuous trajectory experiment in `steeract/trajectory.py`
compares recent log activation levels and their first differences, with bounded
coordinate scaling. It can act on the first observation using its level alone;
later observations add temporal shape. It keeps one decoder direction and
measures candidate alpha responses through the SAE. After the reference horizon,
it compares the current window with the terminal reference window. Prepare a
separate bank with `scripts/prepare_trajectory_development.py`; its
`steering.trajectory` settings must match the saved trajectory bank. This changes
the outcome/support comparison as well as the distance used to choose alpha,
and remains a development experiment until independent validation.

Audit a selected set of completed evaluation configs using
`python -m scripts.audit_evaluation --configs <task-configs...> --required 30 --output <report.json>`.
The audit checks SAE hashes, saved config/bank identity, seed separation, raw
trial labels, valid/invalid accounting, trace cadence, measured descriptor edits,
and summary counts. Use `--required 12` for development audits. The report can
mark the goal achieved only when all nine tasks each have 30 valid trials and
each rate is within five percentage points of its paper target.

## Code Map

| File | Responsibility |
| :--- | :--- |
| [`steeract/config.py`](steeract/config.py) | Recipe validation, task registry, artifact identity |
| [`steeract/data.py`](steeract/data.py) | Activation collection |
| [`steeract/sae.py`](steeract/sae.py) | TopK SAE and training |
| [`steeract/ranking.py`](steeract/ranking.py) | Feature scoring and reference-bank construction |
| [`steeract/method.py`](steeract/method.py) | Temporal descriptor, gates, bounded search, feedback |
| [`steeract/inference.py`](steeract/inference.py) | Hidden-state intervention and evaluation |
| [`steeract/backends/`](steeract/backends/) | π₀.₅ and Fast-WAM integration |

## Citation

If you find SteerAct useful, please cite the [paper](https://arxiv.org/pdf/2610.25632).

```bibtex
@misc{zhou2026steeract,
  title   = {SteerAct: Unlocking Action Models through Few-Shot Temporal Steering},
  author  = {Kaichen Zhou and Zhiyan Li and Chongkai Gao and Yuhan Wang and
             Wenlong Huang and Maggie Wang and Fangneng Zhan and Manling Li and
             Qianqian Wang and Mac Schwager and Zhuang Liu and Li Fei-Fei and
             Mengyu Wang and Paul Pu Liang},
  year    = {2026},
  eprint  = {2610.25632},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url     = {https://arxiv.org/abs/2610.25632}
}
```

---

<p align="center">
  <a href="https://steeract.github.io/">Project Page</a> ·
  <a href="https://steeract.github.io/#method">Method Overview</a> ·
  <a href="https://steeract.github.io/#real-robots">Robot Demonstrations</a>
</p>
