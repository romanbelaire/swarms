"""
Run num_agents scaling for a single fixed conflict baseline.
Writes per-run CSVs plus one aggregate summary CSV.
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

from swarm.config import ENABLED_CONFLICT_ARM_NAMES

_TRAIN_SYMBOLS: tuple[object, object] | None = None


def _get_train_symbols() -> tuple[object, object]:
    global _TRAIN_SYMBOLS
    if _TRAIN_SYMBOLS is None:
        from swarm.training.train import build_parser, run_training

        _TRAIN_SYMBOLS = (build_parser, run_training)
    return _TRAIN_SYMBOLS


def _parse_int_list(raw: str) -> list[int]:
    values = [v.strip() for v in raw.split(",")]
    parsed = [int(v) for v in values if v != ""]
    if len(parsed) == 0:
        raise ValueError("List argument cannot be empty")
    return parsed


def _load_last_row(csv_path: Path) -> dict[str, float]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"Metrics CSV has no rows: {csv_path}")
    return rows[-1]


def _load_last_row_if_complete(
    csv_path: Path,
    expected_episodes: int,
    required_columns: list[str],
) -> dict[str, float] | None:
    if not csv_path.exists():
        return None
    last = _load_last_row(csv_path)
    if "episode" not in last:
        raise ValueError(f"Metrics CSV missing 'episode' column: {csv_path}")
    if int(float(last["episode"])) != expected_episodes:
        return None
    for col in required_columns:
        if col not in last:
            return None
    return last


def _run_task(task: dict) -> tuple[int, dict[str, float]]:
    if task["force_cpu"]:
        os.environ["SWARM_DEVICE"] = "cpu"
    build_parser, run_training = _get_train_symbols()
    base_defaults = build_parser().parse_args([])

    metrics_path = Path(task["out_dir"]) / f"{task['run_name']}.csv"
    required_resume_columns = [
        "episode_env_reward",
        "avg_env_reward_last_50",
        "avg_p_time_percent",
        "avg_c_time_percent",
        "avg_conflict_percent",
        *[f"arm_{name}_prob" for name in ENABLED_CONFLICT_ARM_NAMES],
    ]
    if task["resume"]:
        resumed_last = _load_last_row_if_complete(
            metrics_path,
            int(task["episodes"]),
            required_resume_columns,
        )
        if resumed_last is not None:
            print(f"Skipping completed run {task['run_name']}")
            return int(task["task_idx"]), resumed_last

    run_args = vars(base_defaults).copy()
    run_args["n_agents"] = int(task["n_agents"])
    run_args["episodes"] = int(task["episodes"])
    run_args["max_steps_per_episode"] = int(task["max_steps_per_episode"])
    run_args["num_envs"] = int(task["num_envs"])
    run_args["reward_mode"] = str(task["reward_mode"])
    run_args["mode"] = "task_avoid"
    run_args["baseline_mode"] = "fixed_conflict"
    run_args["fixed_conflict_action"] = str(task["fixed_conflict_action"])
    run_args["bandit_reward_model"] = "neutral_allsame"
    run_args["expert_checkpoint"] = str(task["expert_checkpoint"])
    run_args["metrics_csv"] = str(metrics_path)
    run_args["load_weights"] = None
    run_args["grid_size"] = int(task["grid_size"])
    run_args["num_food"] = int(task["num_food"])
    run_args["local_grid_size"] = int(task["local_grid_size"])

    print(f"Running {task['run_name']}")
    run_training(SimpleNamespace(**run_args))
    return int(task["task_idx"]), _load_last_row(metrics_path)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_conflict_action", type=str, default="move_clear")
    parser.add_argument("--agent_counts", type=str, default="1,2,5,7,10,15,20,25,30,35,40")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_steps_per_episode", type=int, default=200)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--expert_checkpoint", type=str, required=True)
    parser.add_argument("--reward_mode", type=str, default="scenario_mixture", choices=["mean_DR", "scenario_mixture"])
    parser.add_argument("--out_dir", type=str, default="artifacts/ablations")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--num_food", type=int, default=10)
    parser.add_argument("--local_grid_size", type=int, default=5)
    return parser


def main():
    args = build_parser().parse_args()
    if args.cpu:
        os.environ["SWARM_DEVICE"] = "cpu"
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive")

    agent_counts = _parse_int_list(args.agent_counts)
    seeds = _parse_int_list(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    task_idx = 0
    for n_agents in agent_counts:
        for seed in seeds:
            run_name = f"fixed_{args.fixed_conflict_action}_agents{n_agents}_seed{seed}"
            tasks.append(
                {
                    "task_idx": task_idx,
                    "out_dir": str(out_dir),
                    "run_name": run_name,
                    "n_agents": n_agents,
                    "seed": seed,
                    "episodes": args.episodes,
                    "max_steps_per_episode": args.max_steps_per_episode,
                    "num_envs": args.num_envs,
                    "reward_mode": args.reward_mode,
                    "expert_checkpoint": args.expert_checkpoint,
                    "fixed_conflict_action": args.fixed_conflict_action,
                    "resume": args.resume,
                    "force_cpu": args.cpu,
                    "grid_size": args.grid_size,
                    "num_food": args.num_food,
                    "local_grid_size": args.local_grid_size,
                }
            )
            task_idx += 1

    results = _execute_tasks(tasks, args.num_workers)

    rows = []
    for task, (_, last) in zip(tasks, results):
        rows.append(
            {
                "run_name": str(task["run_name"]),
                "method": f"fixed_{args.fixed_conflict_action}",
                "n_agents": int(task["n_agents"]),
                "seed": int(task["seed"]),
                "episode_env_reward": float(last["episode_env_reward"]),
                "avg_env_reward_last_50": float(last["avg_env_reward_last_50"]),
                "avg_p_time_percent": float(last["avg_p_time_percent"]),
                "avg_c_time_percent": float(last["avg_c_time_percent"]),
                "avg_conflict_percent": float(last["avg_conflict_percent"]),
            }
        )

    summary_path = out_dir / f"fixed_{args.fixed_conflict_action}_scaling_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved fixed-scaling summary to {summary_path}")


if __name__ == "__main__":
    main()
