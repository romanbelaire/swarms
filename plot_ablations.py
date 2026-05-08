"""
Plot ablation mean reward versus number of agents.
One subplot is produced per method/bandit in a 3x3 grid.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_summary(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"No rows in summary file: {csv_path}")
    return rows


def _group_stats(rows: list[dict[str, str]], method: str, metric: str):
    buckets: dict[int, list[float]] = {}
    for row in rows:
        if row["method"] != method:
            continue
        n_agents = int(row["n_agents"])
        value = float(row[metric])
        if n_agents not in buckets:
            buckets[n_agents] = []
        buckets[n_agents].append(value)
    xs = sorted(buckets.keys())
    means = [float(np.mean(buckets[x])) for x in xs]
    stds = [float(np.std(buckets[x])) for x in xs]
    return xs, means, stds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, default="artifacts/ablations/ablation_summary.csv")
    parser.add_argument(
        "--fixed_summary_csv",
        type=str,
        default="artifacts/ablations/fixed_conflict_baselines_once.csv",
    )
    parser.add_argument("--out_png", type=str, default="artifacts/ablations/ablation_scaling.png")
    return parser


def main():
    args = build_parser().parse_args()
    rows = _load_summary(Path(args.summary_csv))
    fixed_rows = _load_summary(Path(args.fixed_summary_csv))
    order_pairs = sorted({(int(row["plot_order"]), row["method"]) for row in rows})
    methods = [method for _, method in order_pairs]
    if len(methods) != 9:
        raise ValueError(f"Expected exactly 9 methods for DR reward bandit ablation plot, got {len(methods)}")
    fixed_methods = sorted({row["method"] for row in fixed_rows})

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    metric = "avg_env_reward_last_50"
    fixed_cmap = plt.get_cmap("tab10")
    for idx, method in enumerate(methods):
        ax = axes.ravel()[idx]
        xs, means, stds = _group_stats(rows, method, metric)
        if len(xs) == 0:
            raise ValueError(f"No data found for method: {method}")
        x_arr = np.array(xs, dtype=np.float64)
        mean_arr = np.array(means, dtype=np.float64)
        std_arr = np.array(stds, dtype=np.float64)
        ax.plot(x_arr, mean_arr, marker="o", color="tab:blue")
        ax.fill_between(x_arr, mean_arr - std_arr, mean_arr + std_arr, alpha=0.2, color="tab:blue")
        for fixed_idx, fixed_method in enumerate(fixed_methods):
            fixed_xs, fixed_means, _ = _group_stats(fixed_rows, fixed_method, metric)
            if len(fixed_xs) == 0:
                raise ValueError(f"No data found for fixed baseline method: {fixed_method}")
            fixed_x_arr = np.array(fixed_xs, dtype=np.float64)
            fixed_mean_arr = np.array(fixed_means, dtype=np.float64)
            ax.plot(
                fixed_x_arr,
                fixed_mean_arr,
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
                color=fixed_cmap(fixed_idx % 10),
                label=fixed_method,
            )
        ax.set_title(method)
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel("Mean Reward (last 50)")
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("DR Reward Bandit Ablation with Fixed Baselines: Mean Reward vs Number of Agents", fontsize=14)
    fig.savefig(args.out_png, dpi=150)
    print(f"Saved ablation plot: {args.out_png}")


if __name__ == "__main__":
    main()
