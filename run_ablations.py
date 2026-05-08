"""
Run ablation sweeps across methods, seeds, and agent counts.
Writes per-run metrics CSVs and an aggregate summary CSV.
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

_TRAIN_SYMBOLS: tuple[list[str], object, object] | None = None


def _get_train_symbols() -> tuple[list[str], object, object]:
    global _TRAIN_SYMBOLS
    if _TRAIN_SYMBOLS is None:
        from swarm.training.train import BANDIT_REWARD_MODEL_NAMES, build_parser, run_training

        _TRAIN_SYMBOLS = (BANDIT_REWARD_MODEL_NAMES, build_parser, run_training)
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
    required_columns: list[str] | None = None,
) -> dict[str, float] | None:
    if not csv_path.exists():
        return None
    last = _load_last_row(csv_path)
    if "episode" not in last:
        raise ValueError(f"Metrics CSV missing 'episode' column: {csv_path}")
    if int(float(last["episode"])) != expected_episodes:
        return None
    if required_columns is not None:
        for col in required_columns:
            if col not in last:
                return None
    return last


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
    resume: bool,
    grid_size: int,
    num_food: int,
    local_grid_size: int,
) -> dict[str, float]:
    metrics_path = out_dir / f"{run_name}.csv"
    _, _, run_training = _get_train_symbols()
    required_resume_columns = [
        "episode_env_reward",
        "avg_env_reward_last_50",
        "avg_p_time_percent",
        "avg_c_time_percent",
        "avg_conflict_percent",
        "arm_wait3_prob",
        "arm_backward3_prob",
        "arm_randomwalk3_prob",
        "arm_wait2_forward1_prob",
        "arm_move_clear_prob",
        "arm_handshake_prob",
        "arm_reserve_parity_prob",
        "arm_reserve_parity_escape_prob",
        "arm_priority_swap_n3_prob",
        "arm_pass_food_n3_prob",
    ]
    if resume:
        resumed_last = _load_last_row_if_complete(
            metrics_path,
            episodes,
            required_columns=required_resume_columns,
        )
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


def build_ablation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_counts", type=str, default="3,5,8", help="Comma-separated agent counts")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated integer seeds")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_steps_per_episode", type=int, default=200)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--expert_checkpoint", type=str, required=True)
    parser.add_argument("--reward_mode", type=str, default="scenario_mixture", choices=["mean_DR", "scenario_mixture"])
    parser.add_argument("--out_dir", type=str, default="artifacts/ablations")
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel worker processes for ablation runs")
    parser.add_argument("--resume", action="store_true", help="Skip runs with complete per-run metrics CSV already present")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for all neural network models in every worker")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--num_food", type=int, default=10)
    parser.add_argument("--local_grid_size", type=int, default=5)
    return parser


def main():
    args = build_ablation_parser().parse_args()
    if args.cpu:
        os.environ["SWARM_DEVICE"] = "cpu"
    bandit_reward_model_names, _, _ = _get_train_symbols()
    agent_counts = _parse_int_list(args.agent_counts)
    seeds = _parse_int_list(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        {
            "name": f"dr_{reward_model}",
            "baseline_mode": "bandit_ucb1",
            "bandit_reward_model": reward_model,
            "plot_order": idx,
        }
        for idx, reward_model in enumerate(bandit_reward_model_names)
    ]

    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive")

    fixed_conflict_actions = [
        "wait3",
        "backward3",
        "randomwalk3",
        "wait2_forward1",
        "move_clear",
        "handshake",
        "reserve_parity",
        "reserve_parity_escape",
        "priority_swap_n3",
        "pass_food_n3",
    ]
    fixed_tasks: list[dict[str, str | int]] = []
    fixed_once_rows: list[dict[str, float | str | int]] = []
    fixed_once_seed = seeds[0]
    fixed_task_idx = 0
    for n_agents in agent_counts:
        for action_name in fixed_conflict_actions:
            run_name = f"fixed_once_{action_name}_agents{n_agents}_seed{fixed_once_seed}"
            fixed_tasks.append(
                {
                    "task_idx": fixed_task_idx,
                    "out_dir": str(out_dir),
                    "run_name": run_name,
                    "n_agents": n_agents,
                    "seed": fixed_once_seed,
                    "episodes": args.episodes,
                    "max_steps_per_episode": args.max_steps_per_episode,
                    "num_envs": args.num_envs,
                    "reward_mode": args.reward_mode,
                    "grid_size": args.grid_size,
                    "num_food": args.num_food,
                    "local_grid_size": args.local_grid_size,
                    "expert_checkpoint": args.expert_checkpoint,
                    "baseline_mode": "fixed_conflict",
                    "fixed_conflict_action": action_name,
                    "bandit_reward_model": "neutral_allsame",
                    "resume": args.resume,
                    "force_cpu": args.cpu,
                }
            )
            fixed_task_idx += 1
    fixed_results = _execute_tasks(fixed_tasks, args.num_workers)
    for task, (_, last) in zip(fixed_tasks, fixed_results):
        fixed_once_rows.append(
            {
                "run_name": str(task["run_name"]),
                "method": f"fixed_once_{task['fixed_conflict_action']}",
                "n_agents": int(task["n_agents"]),
                "seed": fixed_once_seed,
                "episode_env_reward": float(last["episode_env_reward"]),
                "avg_env_reward_last_50": float(last["avg_env_reward_last_50"]),
                "avg_p_time_percent": float(last["avg_p_time_percent"]),
                "avg_c_time_percent": float(last["avg_c_time_percent"]),
                "avg_conflict_percent": float(last["avg_conflict_percent"]),
            }
        )

    main_tasks: list[dict[str, str | int]] = []
    task_idx = 0
    summary_rows: list[dict[str, float | str | int]] = []
    for n_agents in agent_counts:
        for seed in seeds:
            for method in methods:
                run_name = f"{method['name']}_agents{n_agents}_seed{seed}"
                main_tasks.append(
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
                        "grid_size": args.grid_size,
                        "num_food": args.num_food,
                        "local_grid_size": args.local_grid_size,
                        "expert_checkpoint": args.expert_checkpoint,
                        "baseline_mode": method["baseline_mode"],
                        "fixed_conflict_action": "backward3",
                        "bandit_reward_model": method["bandit_reward_model"],
                        "method": method["name"],
                        "plot_order": method["plot_order"],
                        "resume": args.resume,
                        "force_cpu": args.cpu,
                    }
                )
                task_idx += 1
    main_results = _execute_tasks(main_tasks, args.num_workers)
    for task, (_, last) in zip(main_tasks, main_results):
        summary_rows.append(
            {
                "run_name": str(task["run_name"]),
                "method": str(task["method"]),
                "plot_order": int(task["plot_order"]),
                "n_agents": int(task["n_agents"]),
                "seed": int(task["seed"]),
                "episode_env_reward": float(last["episode_env_reward"]),
                "avg_env_reward_last_50": float(last["avg_env_reward_last_50"]),
                "avg_p_time_percent": float(last["avg_p_time_percent"]),
                "avg_c_time_percent": float(last["avg_c_time_percent"]),
                "avg_conflict_percent": float(last["avg_conflict_percent"]),
                "arm_wait3_prob": float(last["arm_wait3_prob"]),
                "arm_backward3_prob": float(last["arm_backward3_prob"]),
                "arm_randomwalk3_prob": float(last["arm_randomwalk3_prob"]),
                "arm_wait2_forward1_prob": float(last["arm_wait2_forward1_prob"]),
                "arm_move_clear_prob": float(last["arm_move_clear_prob"]),
                "arm_handshake_prob": float(last["arm_handshake_prob"]),
                "arm_reserve_parity_prob": float(last["arm_reserve_parity_prob"]),
                "arm_reserve_parity_escape_prob": float(last["arm_reserve_parity_escape_prob"]),
                "arm_priority_swap_n3_prob": float(last["arm_priority_swap_n3_prob"]),
                "arm_pass_food_n3_prob": float(last["arm_pass_food_n3_prob"]),
            }
        )

    summary_path = out_dir / "ablation_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved ablation summary to {summary_path}")

    fixed_once_summary_path = out_dir / "fixed_conflict_baselines_once.csv"
    with fixed_once_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fixed_once_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fixed_once_rows)
    print(f"Saved fixed-conflict one-time summary to {fixed_once_summary_path}")


if __name__ == "__main__":
    main()
