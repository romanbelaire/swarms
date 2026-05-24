"""
Run DR bandit ablations with per-arm / arm-relative / step-level credit modes.

Outputs (default root: artifacts/my_role_credit_ablations/):
  <credit_mode>/runs/*.csv
  <credit_mode>/my_role_summary.csv
  <credit_mode>/plots/*.png

Fixed conflict baselines are not re-run by default; use --fixed_summary_csv
to point at an existing fixed_conflict_baselines_once.csv (e.g. from run_ablations.py).
"""

import argparse
import concurrent.futures
import csv
import os
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_ablations import _load_last_row, _load_last_row_if_complete, _parse_int_list
from swarm.config import BANDIT_CONFLICT_ARMS_CSV, ENABLED_CONFLICT_ARM_NAMES

_TRAIN_SYMBOLS = None

CREDIT_TRIALS = ("per_arm_credited", "arm_relative_my_role", "step_level")
OTHERS_SUFFIXES = ("allsame", "allp", "allc")
BANDIT_ARMS = ENABLED_CONFLICT_ARM_NAMES
DEFAULT_OUT_ROOT = "artifacts/my_role_credit_ablations"
DEFAULT_FIXED_SUMMARY_CSV = "artifacts/ablations/fixed_conflict_baselines_once.csv"


def _get_train_symbols():
    global _TRAIN_SYMBOLS
    if _TRAIN_SYMBOLS is None:
        from swarm.training.train import BANDIT_REWARD_MODEL_NAMES, build_parser, run_training

        _TRAIN_SYMBOLS = (BANDIT_REWARD_MODEL_NAMES, build_parser, run_training)
    return _TRAIN_SYMBOLS


def _run_and_collect(
    *,
    base_defaults: argparse.Namespace,
    out_dir: Path,
    run_name: str,
    n_agents: int,
    seed: int,
    episodes: int,
    max_steps_per_episode: int,
    num_envs: int,
    reward_mode: str,
    expert_checkpoint: str,
    baseline_mode: str,
    fixed_conflict_action: str,
    bandit_reward_model: str,
    bandit_conflict_arms: str,
    bandit_credit_mode: str,
    resume: bool,
    grid_size: int,
    num_food: int,
    local_grid_size: int,
) -> dict[str, float]:
    metrics_path = out_dir / f"{run_name}.csv"
    _, _, run_training = _get_train_symbols()
    value_cols = [f"arm_{name}_value" for name in BANDIT_ARMS]
    required_resume_columns = [
        "episode_env_reward",
        "avg_env_reward_last_50",
        *value_cols,
    ]
    if resume:
        resumed_last = _load_last_row_if_complete(
            metrics_path,
            episodes,
            required_columns=required_resume_columns,
        )
        if resumed_last is not None and baseline_mode == "bandit_ucb1":
            csv_credit_mode = resumed_last["bandit_credit_mode"]
            if csv_credit_mode != bandit_credit_mode:
                resumed_last = None
        if resumed_last is not None:
            print(f"Skipping completed run {run_name}")
            return resumed_last
    run_args = vars(base_defaults).copy()
    run_args["n_agents"] = n_agents
    run_args["episodes"] = episodes
    run_args["max_steps_per_episode"] = max_steps_per_episode
    run_args["num_envs"] = num_envs
    run_args["reward_mode"] = reward_mode
    run_args["mode"] = "task_avoid"
    run_args["baseline_mode"] = baseline_mode
    run_args["fixed_conflict_action"] = fixed_conflict_action
    run_args["bandit_reward_model"] = bandit_reward_model
    run_args["bandit_conflict_arms"] = bandit_conflict_arms
    run_args["bandit_credit_mode"] = bandit_credit_mode
    run_args["expert_checkpoint"] = expert_checkpoint
    run_args["metrics_csv"] = str(metrics_path)
    run_args["load_weights"] = None
    run_args["grid_size"] = grid_size
    run_args["num_food"] = num_food
    run_args["local_grid_size"] = local_grid_size
    print(f"Running {run_name}")
    run_training(SimpleNamespace(**run_args))
    return _load_last_row(metrics_path)


def _run_task(task: dict) -> tuple[int, dict[str, float]]:
    if task["force_cpu"]:
        os.environ["SWARM_DEVICE"] = "cpu"
    _, build_parser, _ = _get_train_symbols()
    base_defaults = build_parser().parse_args([])
    last = _run_and_collect(
        base_defaults=base_defaults,
        out_dir=Path(task["out_dir"]),
        run_name=task["run_name"],
        n_agents=task["n_agents"],
        seed=task["seed"],
        episodes=task["episodes"],
        max_steps_per_episode=task["max_steps_per_episode"],
        num_envs=task["num_envs"],
        reward_mode=task["reward_mode"],
        expert_checkpoint=task["expert_checkpoint"],
        baseline_mode=task["baseline_mode"],
        fixed_conflict_action=task["fixed_conflict_action"],
        bandit_reward_model=task["bandit_reward_model"],
        bandit_conflict_arms=task["bandit_conflict_arms"],
        bandit_credit_mode=task["bandit_credit_mode"],
        resume=task["resume"],
        grid_size=task["grid_size"],
        num_food=task["num_food"],
        local_grid_size=task["local_grid_size"],
    )
    return int(task["task_idx"]), last


def _execute_tasks(tasks: list[dict], num_workers: int) -> list[tuple[int, dict[str, float]]]:
    if num_workers <= 1:
        return [_run_task(task) for task in tasks]
    results: list[tuple[int, dict[str, float]]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_run_task, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item[0])
    return results


def _split_reward_model(reward_model: str) -> tuple[str, str]:
    for suffix in OTHERS_SUFFIXES:
        token = f"_{suffix}"
        if reward_model.endswith(token):
            my_role = reward_model[: -len(token)]
            return my_role, suffix
    raise ValueError(f"Cannot parse reward model: {reward_model}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_counts", type=str, default="1,10,20,40,60")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_steps_per_episode", type=int, default=200)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--expert_checkpoint", type=str, required=True)
    parser.add_argument("--reward_mode", type=str, default="scenario_mixture", choices=["mean_DR", "scenario_mixture"])
    parser.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--fixed_summary_csv",
        type=str,
        default=DEFAULT_FIXED_SUMMARY_CSV,
        help="Existing fixed-conflict baseline summary for rank-correlation vs env reward plots.",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--credit_modes",
        type=str,
        default=",".join(CREDIT_TRIALS),
        help="Comma-separated subset of per_arm_credited, arm_relative_my_role, step_level",
    )
    parser.add_argument(
        "--bandit_conflict_arms",
        type=str,
        default=BANDIT_CONFLICT_ARMS_CSV,
    )
    parser.add_argument(
        "--run_fixed",
        action="store_true",
        help="Re-run fixed conflict baselines into <out_root>/fixed/runs/ (off by default).",
    )
    parser.add_argument("--skip_bandit", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.cpu:
        os.environ["SWARM_DEVICE"] = "cpu"
    bandit_reward_model_names, _, _ = _get_train_symbols()
    agent_counts = _parse_int_list(args.agent_counts)
    seeds = _parse_int_list(args.seeds)
    credit_modes = [v.strip() for v in args.credit_modes.split(",") if v.strip()]
    for mode in credit_modes:
        if mode not in CREDIT_TRIALS:
            raise ValueError(f"Unknown credit mode {mode}; expected one of {CREDIT_TRIALS}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    fixed_summary_path = Path(args.fixed_summary_csv)
    fixed_conflict_actions = list(BANDIT_ARMS)

    if args.run_fixed:
        fixed_dir = out_root / "fixed" / "runs"
        fixed_dir.mkdir(parents=True, exist_ok=True)
        fixed_once_seed = seeds[0]
        fixed_tasks: list[dict] = []
        fixed_task_idx = 0
        for n_agents in agent_counts:
            for action_name in fixed_conflict_actions:
                run_name = f"fixed_once_{action_name}_agents{n_agents}_seed{fixed_once_seed}"
                fixed_tasks.append(
                    {
                        "task_idx": fixed_task_idx,
                        "out_dir": str(fixed_dir),
                        "run_name": run_name,
                        "n_agents": n_agents,
                        "seed": fixed_once_seed,
                        "episodes": args.episodes,
                        "max_steps_per_episode": args.max_steps_per_episode,
                        "num_envs": args.num_envs,
                        "reward_mode": args.reward_mode,
                        "grid_size": 10,
                        "num_food": 10,
                        "local_grid_size": 5,
                        "expert_checkpoint": args.expert_checkpoint,
                        "baseline_mode": "fixed_conflict",
                        "fixed_conflict_action": action_name,
                        "bandit_reward_model": "neutral_allsame",
                        "bandit_conflict_arms": "",
                        "bandit_credit_mode": "episode_shared",
                        "resume": args.resume,
                        "force_cpu": args.cpu,
                    }
                )
                fixed_task_idx += 1
        fixed_results = _execute_tasks(fixed_tasks, args.num_workers)
        fixed_rows = []
        for task, (_, last) in zip(fixed_tasks, fixed_results):
            fixed_rows.append(
                {
                    "run_name": str(task["run_name"]),
                    "method": f"fixed_once_{task['fixed_conflict_action']}",
                    "arm": str(task["fixed_conflict_action"]),
                    "n_agents": int(task["n_agents"]),
                    "seed": fixed_once_seed,
                    "avg_env_reward_last_50": float(last["avg_env_reward_last_50"]),
                }
            )
        written_fixed_summary = out_root / "fixed" / "fixed_conflict_baselines_once.csv"
        written_fixed_summary.parent.mkdir(parents=True, exist_ok=True)
        with written_fixed_summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fixed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(fixed_rows)
        print(f"Saved {written_fixed_summary}")
        fixed_summary_path = written_fixed_summary

    if args.skip_bandit:
        if not args.skip_plots:
            from plot_my_role_rank_corr import plot_all_credit_modes

            plot_all_credit_modes(out_root, credit_modes, BANDIT_ARMS, fixed_summary_path)
        return

    for credit_mode in credit_modes:
        trial_dir = out_root / credit_mode
        runs_dir = trial_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        main_tasks: list[dict] = []
        task_idx = 0
        summary_rows: list[dict] = []
        for n_agents in agent_counts:
            for seed in seeds:
                for reward_model in bandit_reward_model_names:
                    my_role, others_role = _split_reward_model(reward_model)
                    run_name = f"dr_{reward_model}_{credit_mode}_agents{n_agents}_seed{seed}"
                    main_tasks.append(
                        {
                            "task_idx": task_idx,
                            "out_dir": str(runs_dir),
                            "run_name": run_name,
                            "n_agents": n_agents,
                            "seed": seed,
                            "episodes": args.episodes,
                            "max_steps_per_episode": args.max_steps_per_episode,
                            "num_envs": args.num_envs,
                            "reward_mode": args.reward_mode,
                            "grid_size": 10,
                            "num_food": 10,
                            "local_grid_size": 5,
                            "expert_checkpoint": args.expert_checkpoint,
                            "baseline_mode": "bandit_ucb1",
                            "fixed_conflict_action": "backwards2",
                            "bandit_reward_model": reward_model,
                            "bandit_conflict_arms": args.bandit_conflict_arms,
                            "bandit_credit_mode": credit_mode,
                            "resume": args.resume,
                            "force_cpu": args.cpu,
                            "my_role": my_role,
                            "others_role": others_role,
                            "reward_model": reward_model,
                        }
                    )
                    task_idx += 1
        main_results = _execute_tasks(main_tasks, args.num_workers)
        for task, (_, last) in zip(main_tasks, main_results):
            row = {
                "run_name": str(task["run_name"]),
                "method": f"dr_{task['reward_model']}_{credit_mode}",
                "bandit_credit_mode": credit_mode,
                "bandit_reward_model": str(task["reward_model"]),
                "my_role": str(task["my_role"]),
                "others_role": str(task["others_role"]),
                "n_agents": int(task["n_agents"]),
                "seed": int(task["seed"]),
                "avg_env_reward_last_50": float(last["avg_env_reward_last_50"]),
            }
            for arm in BANDIT_ARMS:
                row[f"arm_{arm}_value"] = float(last[f"arm_{arm}_value"])
            summary_rows.append(row)
        summary_path = trial_dir / "my_role_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved {summary_path}")

    if not args.skip_plots:
        from plot_my_role_rank_corr import plot_all_credit_modes

        plot_all_credit_modes(out_root, credit_modes, BANDIT_ARMS, fixed_summary_path)


if __name__ == "__main__":
    main()
