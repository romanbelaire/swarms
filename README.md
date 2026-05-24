# Swarm Research Codebase

This repository contains DQN-based experiments for rational swarm foraging with difference-reward shaping.

The codebase is organized for readability and sharing:
- reusable code in `src/swarm`
- runnable CLIs as `python src/swarm/...py` (and root ablation utilities)
- generated outputs in `artifacts`
- math/spec docs in `docs`

## Repository Layout

- `src/swarm/env`: environment implementation
- `src/swarm/agents`: DQN, critics, replay buffers, frozen expert
- `src/swarm/training`: training pipeline
- `src/swarm/training/train.py`: main DQN / DR training CLI
- `src/swarm/training/train_expert.py`: expert DQN on base env (sparse `env.step` rewards)
- `src/swarm/evaluate.py`: headless evaluation CLI
- `src/swarm/play.py`: pygame visualization CLI
- `src/swarm/plotting.py`: convergence plotting CLI
- `run_ablations.py`, `plot_ablations.py`: ablation sweep and scaling plot (repo root)

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Quickstart

Train the expert policy: `src/swarm/training/train_expert.py` fits a 5-action DQN on the **sparse PettingZoo rewards** from `env.step` (food deliveries). 

```powershell
.\venv\Scripts\python.exe .\src\swarm\training\train_expert.py --n_agents 1 --episodes 2000 --out .\artifacts\expert\dqn_weights_agent_0.pt
```

Train task-avoid mode (with pretrained expert):

```powershell
.\venv\Scripts\python.exe .\src\swarm\training\train.py --mode task_avoid --n_agents 5 --episodes 2000 --expert_checkpoint .\artifacts\expert\dqn_weights_agent_0.pt
```

Evaluate headless:

```powershell
.\venv\Scripts\python.exe .\src\swarm\evaluate.py --n_agents 5 --episodes 10 --max_steps_per_episode 500
```

Visualize rollout:

```powershell
.\venv\Scripts\python.exe .\src\swarm\play.py --n_agents 5 --grid_size 7 --num_food 5 --max_steps 500 --step_delay 0.05
```

For agents trained with `--reward_mode scenario_mixture`, the same run shows a DR side panel (gates `u`, `v`, joint `w`, mixed reward) if `dr_mixture_agent_*.pt` and `dr_mixture_meta.json` exist under `--weights_dir` (or `--dr_weights_dir`). Use `--no-show_dr` for the original grid-only window.

Plot convergence:

```powershell
.\venv\Scripts\python.exe .\src\swarm\plotting.py --metrics_csv .\artifacts\training_metrics.csv --out_png .\artifacts\training_convergence.png
```

## Output Artifacts

Default outputs are written to `artifacts/`:
- `training_metrics.csv`
- `training_convergence.png`
- `dqn_weights_agent_*.pt`
- `pc_critic_agent_*.pt`
- `expert/dqn_weights_agent_0.pt` and `expert/expert_training_metrics.csv` when using `src/swarm/training/train_expert.py`

## Common Flags

- `--reward_mode`: `mean_DR` or `scenario_mixture`
- `--num_envs`: number of synchronous environment replicas
- `--alpha`, `--gamma`: DQN optimizer settings
- `--critic_lr`, `--critic_gamma`: critic optimizer settings
- `--dr_entropy_coef`: entropy regularization for DR gates
- `--baseline_mode`: `none`, `bandit_ucb1`, `random_conflict`, `fixed_conflict`
- `--fixed_conflict_action`: one of `ENABLED_CONFLICT_ARM_NAMES` in `src/swarm/config.py` (`freeze_tag`, `randomwalk3`, `wait3`, `move_clear`, `backwards2`)
- Task-avoid high-level actions are enabled via `TASK_AVOID_ENABLED_ACTION_IDS` in `src/swarm/config.py`; set an action constant to `-1` to disable it.
- Bandit / ablation conflict arms default to `BANDIT_CONFLICT_ARMS_CSV` in `src/swarm/config.py` (same five macros).

## Ablation Baselines

Run a non-learned (or bandit-only) baseline:

```powershell
.\venv\Scripts\python.exe .\src\swarm\training\train.py --baseline_mode random_conflict --expert_checkpoint .\artifacts\expert\dqn_weights_agent_0.pt --episodes 200
```

Run UCB-only scaling ablations over 9 DR reward models, seeds, and robot counts:

```powershell
.\venv\Scripts\python.exe .\run_ablations.py --expert_checkpoint .\artifacts\expert\dqn_weights_agent_0.pt --agent_counts 3,5,8 --seeds 0,1,2 --episodes 200 --num_workers 4
```

For the DR reward 9-bandit ablation (UCB only; each subplot uses a different DR reward model) on CPU: `$env:SWARM_DEVICE="cpu"; .\venv\Scripts\python.exe .\run_ablations.py --expert_checkpoint .\artifacts\expert\dqn_weights_agent_0.pt --agent_counts 3,5,8 --seeds 0,1,2 --episodes 200 --reward_mode scenario_mixture`

Plot ablation scaling:

```powershell
.\venv\Scripts\python.exe .\plot_ablations.py --summary_csv .\artifacts\ablations\ablation_summary.csv --out_png .\artifacts\ablations\ablation_scaling.png
```

Additional ablation outputs:
- per-run CSVs under `artifacts/ablations/`
- summary CSV: `artifacts/ablations/ablation_summary.csv`
- scaling plot: `artifacts/ablations/ablation_scaling.png`

## Contributing Notes

- **Device:** PyTorch models and batches use **CUDA when available** (else Apple **MPS**, else **CPU**). Force CPU with `set SWARM_DEVICE=cpu` (PowerShell: `$env:SWARM_DEVICE="cpu"`). GPU runs are not bitwise-deterministic unless you add your own `torch.use_deterministic_algorithms` / CUDNN settings.
- Preserve fail-fast behavior: raise explicit errors for invalid states.
- Prefer changes in `src/swarm/*`. Run CLIs with `python src/swarm/...py` from the repo root (those modules insert `src` on `sys.path` when executed as scripts).
