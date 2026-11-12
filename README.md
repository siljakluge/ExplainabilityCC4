# ExplainabilityCC4

**Training, evaluation, robustness analysis, and explainability for autonomous cyber-defense agents in CAGE Challenge 4.**

This repository contains a locally modified version of the **CybORG CAGE Challenge 4 (CC4)** environment, multi-agent reinforcement-learning and heuristic blue agents, six red-attacker profiles, evaluation scripts, explanation utilities, and selected result artifacts.

The research context is the paper **“Robustness Evaluation of Autonomous Cyber Defense Agents under Adversarial Strategy Variation”** (IEEE CSR 2026). The repository provides the implementation and selected outputs relevant to that work; **the published experiments have not been fully reproduced from a fresh clone as part of this release**.

## Scope

The project supports:

- **Blue agents:** graph-based PPO (GNN / MARL), a rule-based heuristic, and a Sleep baseline.
- **Red strategies:** `fsm_default`, `stealth_pivot`, `lateral_spread`, `impact_rush`, `deception_aware`, and `discovery`.
- **Training:** single-profile, weighted-mixture, and curriculum-based training.
- **Evaluation:** fixed-strategy runs, multi-strategy sweeps, and a weighted-mixture run.
- **Analysis:** episodic rewards, red-strategy robustness scores, action frequencies, reward decomposition, and SHAP-based surrogate explanations.

The environment simulates multiple defending blue agents under partial observability, with agent communication and phase-dependent mission rewards. All experiments in this repository use a **simulation**, not a production cyber-defense deployment.

## Repository layout

| Path | Purpose |
| --- | --- |
| `CybORG/` | Bundled, locally modified CC4 environment, red agents, reward logic, and upstream-derived tests |
| `Heuristic/` | Rule-based blue-agent implementations |
| `models/` | GNN-PPO actor/critic and memory management |
| `wrappers/` | Graph observation and heuristic environment adapters |
| `submission.py` | Selects the **one** currently active blue-agent implementation |
| `train.py` | GNN-PPO training (single, mixture, or weighted curriculum) |
| `train_curriculum.py` | Sequential red-profile curriculum training |
| `evaluate.py` | Episodic evaluation and attacker-profile sweep |
| `explain.py` | Reward/action logging and optional SHAP analysis |
| `SHAP/` | Surrogate-model SHAP implementations |
| `plotting/` | Training, action-frequency, and reward-decomposition plotting scripts |
| `tables/` | Robustness-score and table-generation utilities |
| `weights/contractor_active/` | Five provided GNN model-weight files (`gnn_ppo-0.pt` to `gnn_ppo-4.pt`) |
| `checkpoints/` | Five provided `contractoractive` training checkpoints |
| `logs/`, `exported_csv/` | Selected training logs and processed CSV data |
| `Results/` | Selected published/archived figures and aggregated robustness scores |
| `create_csv_logs.py`, `export_weights.py` | Log conversion and model export utilities |
| `old/` | Historical implementations; not canonical entry points |

**Important:** `Results/` contains selected existing artifacts. It is **not** a complete set of the per-episode evaluation logs needed to regenerate every paper result.

## Requirements and installation

- **Python 3.11** is recommended.
- Use a fresh virtual environment and install the packages specified in `requirements.txt`.
- Run commands from the **repository root** so that local imports such as `from CybORG ...` resolve to the bundled code.
- A CPU is sufficient for basic evaluation; training may benefit from CUDA or Apple Silicon acceleration. The training scripts select CUDA, MPS, or CPU based on availability.

### macOS / Linux

```bash
git clone https://github.com/siljakluge/ExplainabilityCC4.git
cd ExplainabilityCC4
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### Windows (PowerShell)

```powershell
git clone https://github.com/siljakluge/ExplainabilityCC4.git
cd ExplainabilityCC4
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

**Use the bundled `CybORG/` source.** Do not separately install the unmodified upstream `cage-challenge-4` package into the same environment; this project contains modifications to attacker behavior and rewards that may not exist upstream.

### Installation and import smoke test

```bash
python -c "import torch, torch_geometric, gymnasium, ray.rllib, shap, CybORG; print('Core imports OK')"
python -c "from submission import Submission; s = Submission(); print(s.NAME, sorted(s.AGENTS))"
```

The second command also attempts to load the currently selected agents and their weights. A successful import is **not** a substitute for an end-to-end evaluation. For the bundled CC4 tests, run:

```bash
python -m pytest CybORG/Tests/test_cc4 -q
```

## Select a blue agent

**`submission.py` is the model-selection entry point.** At the currently documented repository revision, the **GNN** `Submission` class is active. It loads these five files:

```text
weights/contractor_active/gnn_ppo-0.pt
weights/contractor_active/gnn_ppo-1.pt
weights/contractor_active/gnn_ppo-2.pt
weights/contractor_active/gnn_ppo-3.pt
weights/contractor_active/gnn_ppo-4.pt
```

The heuristic and Sleep implementations are also defined in `submission.py` but are inside commented-out code blocks. **There is currently no command-line `--agent` selector**: to evaluate a different baseline, edit `submission.py` so that exactly one `Submission` class is active, and use the matching wrapper:

| Blue agent | Agent implementation | Wrapper | Explanation flag |
| --- | --- | --- | --- |
| GNN | Loaded `InductiveGraphPPOAgent` weights | `GraphWrapper` | `--non-heuristic` |
| Heuristic | `Heuristic.Heuristic_Agent_v0_3.H_Agent` | `HeuristicWrapper` | `--is-heuristic` |
| Sleep | `SleepAgent` | `BlueFixedActionWrapper` | `--non-heuristic` (without SHAP recommended) |

The `--model-name` argument in `evaluate.py` **names the output directory**; it does **not** change the policy loaded by `submission.py`. Check the active `Submission.NAME` before every comparative run. The exact correspondence between a published model configuration and a checkpoint should be documented separately in an experiment manifest before formal reproduction claims.

## Quick start: evaluate the supplied GNN

Start with a small, single-profile run to check that environment construction, agent loading, stepping, and output writing succeed:

```bash
python evaluate.py \
  --mode single \
  --strategy single \
  --single_profile fsm_default \
  --max-eps 1 \
  --distribute 1 \
  --seed 1337 \
  --submission-path . \
  --output-path evals/GNN/smoke
```

For a broader evaluation, the script supports a sweep over all six registered attacker profiles **plus a mixed-profile run**:

```bash
python evaluate.py \
  --mode sweep \
  --model-name GNN \
  --eps-per-profile 100 \
  --distribute 1 \
  --seed 1337 \
  --submission-path . \
  --out-root evals
```

`evaluate.py` currently uses **500 steps per episode** (`EPISODE_LENGTH = 500`). With six profiles and one mixture run, the command above requests up to **700 episodes**. The primary outputs are under `evals/GNN/<profile>/` and include `summary.json`, `summary.txt`, `scores.txt`, and optional full action/observation dumps.

**Caution:** `evaluate.py` currently parses reward-mode CLI options but does not propagate them into its evaluation execution. Do not assume `--phase_reward_mode` or `--reward_mode_strategy` changes an evaluation until the script is fixed and verified. The bundled reward machine also writes JSONL logs to a configurable path (default: `reward_log.jsonl`), so parallel workers should not share a single output file without a safe logging strategy.

## Explainability and reward/action logging

`explain.py` runs episodes under individual attacker profiles and writes structured logs. Its `--mode sweep` executes the registered profiles (excluding the verbose debugging variant); `--mode single` can use a fixed profile or sampled mixture.

A short smoke run for the active GNN is:

```bash
python explain.py \
  --mode sweep \
  --max-eps 1 \
  --episode-length 50 \
  --seed 1337 \
  --phase_reward_mode default \
  --non-heuristic \
  --submission-path . \
  --output ResultsRuns
```

For SHAP-based analysis, add `--shap` and use enough episodes to obtain a usable dataset. For the heuristic, activate its `Submission` implementation and use `--is-heuristic --shap` instead.

The explanation scripts use **surrogate classifiers** to analyze action behavior: a random-forest surrogate for the GNN path and a gradient-boosting surrogate for the heuristic path. Consequently, their SHAP values explain the fitted **surrogate predictions**, not direct gradients or intrinsic attributions of the original PPO policy. Consult surrogate predictive metrics when interpreting explanations.

`explain.py` supports reward modes:

- `default`: the configured phase-specific CC4 rewards;
- `contractor_off`: removes contractor-subnet reward contributions;
- `red_only`: uses red-action-centered reward components;
- `--reward_blue`: enables the optional blue Analyse reward (as implemented by the bundled reward machine).

For example, to compare the three modes using the current policy:

```bash
python explain.py \
  --mode sweep \
  --max-eps 10 \
  --reward_mode_strategy sweep \
  --reward_modes default,contractor_off,red_only \
  --non-heuristic \
  --submission-path . \
  --output ResultsRuns
```

Outputs are grouped under `ResultsRuns/<Submission.NAME>_<timestamp>/<reward_mode>/<attacker>/` and can include `summary_scalar.json`, `reward_log.jsonl`, `scalar_rewards.jsonl`, `actions.jsonl`, and `SHAPAnalysis/` if enabled.

## Training and weight export

Training generates new policies; it is **not required** to evaluate the supplied GNN weights.

### Single-profile or mixture PPO training

```bash
# Development configuration; still performs multiple training episodes
python train.py example_single --debug --strategy single --single_profile fsm_default

# Weighted red-strategy mixture
python train.py example_mix \
  --strategy mixture \
  --profile_weights "fsm_default=0.35,stealth_pivot=0.25,lateral_spread=0.20,impact_rush=0.15,deception_aware=0.05"
```

`train.py` also supports `--strategy curriculum`, `--warmup_updates`, `--harden_updates`, `--KIServer`, and `--log_per_profile`. The `--KIServer` switch changes training hyperparameters; it does **not** configure a network connection or guarantee a GPU.

### Sequential curriculum

```bash
python train_curriculum.py example_curriculum \
  --debug \
  --curriculum_order fsm_default,stealth_pivot,lateral_spread
```

Training saves checkpoint files as `checkpoints/<run_name>-<agent_index>_checkpoint.pt` and training histories under `logs/<run_name>.pt`.

### Export the existing contractor-active checkpoints

```bash
python export_weights.py
```

At the documented revision, `export_weights.py` reads `checkpoints/contractoractive-{0..4}_checkpoint.pt` and writes `weights/contractor_active/gnn_ppo-{0..4}.pt`. **These output files already exist; running the export overwrites them.** Preserve the supplied weights or use a separate destination when comparing artifacts.

## Analysis artifacts and reproduction status

The repository includes selected plots in `Results/ActionPlots/` and `Results/Reward_Decomp/` and the following aggregate robustness results in `Results/Robustness/robustness_scores.csv` and `.json`:

| Blue policy | Robustness score |
| --- | ---: |
| Heuristic | -221.84 |
| SimpleGNN | -324.75 |
| Sleep | -5827.81 |

The robustness metric implemented in `tables/robustness_scores.py` is:

$$
R(b) = \overline{\mu}_b - \lambda_1 \overline{\sigma}_b - \lambda_2\,\mathrm{Std}_{r}(\mu_{b,r})
$$

where the first term is mean performance across attacker strategies, the second penalizes within-strategy variability, and the last penalizes variability across strategies. The script uses `lambda_1 = lambda_2 = 1.0` and the six strategies listed above.

**Reproduction boundary:** The stored CSV/JSON provide final aggregate values, not the full episode-level inputs. As currently written, `tables/robustness_scores.py` expects `Result/<Model>/<Strategy>/summary_scalar.json`, including a `RedVariants` model; those inputs are not supplied in this repository. The published values therefore cannot be recomputed end to end from the checked-in results without restoring the inputs and reconciling the script configuration.

### Training curves

A training-history CSV can be generated from a corresponding `.pt` log, for example:

```bash
python create_csv_logs.py --logs logs/default_curriculum.pt --outdir exported_csv
python -c "from pathlib import Path; Path('Plots').mkdir(exist_ok=True)"
python plotting/plot_training.py
```

`plot_training.py` currently expects `exported_csv/default_curriculum.csv`. The plotting scripts for action frequencies and reward decomposition currently reference **`Result/` (singular)** and hard-coded model folders; this does not directly match the shipped `Results/` output-artifact layout. Review/update the configured paths before rerunning those plots. The checked-in figures can be viewed independently of those scripts.

## Known limitations and validation checklist

The following are **known limitations of the current code**, not assurances of a successful release test:

1. **Reward-mode handling in evaluation:** reward-mode CLI parameters in `evaluate.py` are parsed but not applied to the simulation runs.
2. **Reward flags in training:** `train.py` and `train_curriculum.py` currently write `CYBORG_REWARD_blue`, while `CybORG/Shared/BlueRewardMachine.py` reads `CYBORG_REWARD_BLUE`; the `--reward_blue` flag therefore requires a code fix for training.
3. **Reward accounting:** `evaluate.py` exits on a terminal step before appending that step's reward, whereas `explain.py` includes it. Align the two before making numerical comparisons.
4. **Result regeneration:** complete per-episode evaluation inputs and an experiment-to-checkpoint manifest are not included. The stored robustness table and figures are reference artifacts, not proof of end-to-end reproducibility.
5. **Plot paths:** some plotting and table scripts target `Result/` while the checked-in figures live in `Results/` and fresh explanation runs use a timestamped path.
6. **Version compatibility:** the pinned Python dependencies have not been verified by a clean, cross-platform CI installation; validate model loading and the full CC4 test suite on the target system.
7. **Checkpoint trust:** `.pt` files should only be loaded from trusted sources. Review PyTorch serialization behavior when upgrading or changing environments.

Before declaring a client-ready release, the receiving team should confirm that:

- `python -m pip check` succeeds in a fresh Python 3.11 environment;
- `python -m pytest CybORG/Tests/test_cc4 -q` passes;
- all five policy weights load and a one-episode evaluation completes;
- the heuristic, GNN, and Sleep baselines can each be evaluated under all six red strategies with consistent reward settings;
- paper tables and figures can be rebuilt from archived experiment inputs;
- a fixed Git commit/tag, runtime information, checkpoint hashes, and original experiment configuration are recorded.

## Source provenance and licensing

The `CybORG/` directory includes code derived from [CAGE Challenge 4](https://github.com/cage-challenge/cage-challenge-4) and modifications made for this research. Consult upstream notices and the individual source files for applicable attribution and license obligations.

**No top-level license file is currently supplied in this repository.** Public GitHub visibility is not itself a grant of redistribution or commercial-use rights. Any external/customer handover must be covered by the relevant project-owner approval, third-party license compliance, and the applicable contractual terms.

## Citation

If this repository is used in research, cite the associated paper:

*“Robustness Evaluation of Autonomous Cyber Defense Agents under Adversarial Strategy Variation,” IEEE International Conference on Cyber Security and Resilience (CSR), 2026.*

For an archival release, add the full author list, DOI/proceedings URL, the exact Git release tag, and a mapping between published experiments and the provided checkpoints.
