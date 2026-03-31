# ExplainabilityCC4

This repository contains code for training, evaluating, and analyzing autonomous cyber defense agents in the CAGE Challenge 4 environment. It includes reinforcement learning agents, heuristic baselines, explainability scripts, evaluation pipelines, weight export utilities, log conversion scripts, and plotting tools for robustness, reward decomposition, and action analysis.

## Project Overview

The project focuses on:
- training graph-based PPO agents for CAGE Challenge 4
- evaluating models against different red attacker profiles
- analyzing reward decomposition and action behavior
- generating explainability outputs such as SHAP-based analyses
- exporting model checkpoints into submission-ready weights
- converting training logs into CSV files for plotting
- comparing learned agents with heuristic baselines

## Repository Structure

```text
ExplainabilityCC4/
├── .venv/
├── checkpoints/
├── CybORG/
├── evals/
├── exported_csv/
├── Heuristic/
├── logs/
├── models/
├── old/
├── plotting/
├── Result/
├── Results/
├── reward_plots/
├── SHAP/
├── tables/
├── Users/
├── weights/
├── wrappers/
├── config.json
├── create_csv_logs.py
├── evaluate.py
├── explain.py
├── export_weights.py
├── submission.py
├── train.py
├── train_curriculum.py
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

---

## Main Scripts

### `train.py`

Train PPO-based agents.

```bash
python train.py red_rew_only \
  --KIServer \
  --phase_reward_mode red_only \
  --strategy single \
  --single_profile fsm_default
```

---

### `train_curriculum.py`

Sequential curriculum training over attacker profiles.

```bash
python train_curriculum.py curriculum_run \
  --KIServer \
  --stage_episodes 5000 \
  --curriculum_order fsm_default,stealth_pivot,lateral_spread
```

---

### `evaluate.py`

Evaluate a submission.

```bash
python evaluate.py \
  --mode sweep \
  --model-name simplegnn \
  --eps-per-profile 100 \
  --submission-path .
```

---

### `explain.py`

Run explainability and reward analysis.

```bash
python explain.py \
  --mode sweep \
  --max-eps 10 \
  --non-heuristic \
  --phase_reward_mode red_only \
  --submission-path .
```

---

## Plotting

### `plot_actions.py`

Plots action frequencies.

```bash
python plotting/plot_actions.py \
  --out_dir Plots/ActionPlots \
  --top_k 25
```

---

### `plot_rew_decomp.py`

Plots reward decomposition.

```bash
python plotting/plot_rew_decomp.py --out Plots/RewardDecomp
```

---

### `plot_training.py`

Plots training curves from CSV logs.

```bash
python plotting/plot_training.py
```

Requires CSVs in `exported_csv/`.

---

## Utilities

### `create_csv_logs.py`

Convert `.pt` logs → CSV.

```bash
python create_csv_logs.py \
  --logs logs/exp_mix.pt \
  --outdir exported_csv
```

---

### `export_weights.py`

Export checkpoints → submission weights.

```bash
python export_weights.py
```

---

## Submission

### Learned PPO model

```python
class Submission:
    NAME = "RedVariants"

    def __init__(self):
        self.AGENTS = {
            f"blue_agent_{i}": load(
                f"weights/exp_mix/gnn_ppo-{i}.pt",
                map_location="cpu"
            )
            for i in range(5)
        }

    @staticmethod
    def wrap(env):
        return GraphWrapper(env)
```

### Baseline (Sleep)

```python
class Submission:
    NAME = "Sleep"

    def __init__(self):
        self.AGENTS = {
            f"blue_agent_{i}": SleepAgent(agent_name=f"blue_agent_{i}")
            for i in range(5)
        }

    @staticmethod
    def wrap(env):
        return BlueFixedActionWrapper(env)
```

---

## Workflow

```bash
# 1. Train
python train.py my_run --KIServer

# 2. Export weights
python export_weights.py

# 3. Evaluate
python evaluate.py --mode sweep --submission-path .

# 4. Explain
python explain.py --mode sweep --submission-path .

# 5. Convert logs
python create_csv_logs.py --logs logs/my_run.pt

# 6. Plot
python plotting/plot_training.py
python plotting/plot_actions.py
python plotting/plot_rew_decomp.py
```

---

## Outputs

- `logs/` → training logs  
- `checkpoints/` → checkpoints  
- `weights/` → exported models  
- `evals/` → evaluation results  
- `Results/` → explainability outputs  
- `exported_csv/` → processed logs  
- `Plots/` → figures  

---

## Notes

- `submission.py` must match the model you want to evaluate  
- run `create_csv_logs.py` before plotting training curves  
- run `export_weights.py` before evaluation  
- reward settings can be changed via:
  - `--phase_reward_mode`
  - `--reward_blue`



