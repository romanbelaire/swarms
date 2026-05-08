"""
Plot num-agent scaling for a single fixed conflict baseline summary CSV.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"No rows in summary file: {csv_path}")
    return rows


def _group_metric(rows: list[dict[str, str]], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    buckets: dict[int, list[float]] = {}
    for row in rows:
        n_agents = int(row["n_agents"])
        value = float(row[metric])
        if n_agents not in buckets:
            buckets[n_agents] = []
        buckets[n_agents].append(value)
    xs = np.array(sorted(buckets.keys()), dtype=np.float64)
    means = np.array([float(np.mean(buckets[int(x)])) for x in xs], dtype=np.float64)
    stds = np.array([float(np.std(buckets[int(x)])) for x in xs], dtype=np.float64)
    return xs, means, stds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="artifacts/ablations/fixed_reserve_parity_escape_scaling_summary.csv",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="artifacts/ablations/fixed_reserve_parity_escape_scaling.png",
    )
    return parser


def _plot_series(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    ylabel: str,
):
    xs, means, stds = _group_metric(rows, metric)
    ax.plot(xs, means, marker="o")
    ax.fill_between(xs, means - stds, means + stds, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel(ylabel)


def main():
    args = build_parser().parse_args()
    rows = _load_rows(Path(args.summary_csv))
    method_names = sorted({row["method"] for row in rows})
    if len(method_names) != 1:
        raise ValueError(f"Expected exactly one method in summary CSV, got {len(method_names)}: {method_names}")
    method_name = method_names[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _plot_series(axes[0, 0], rows, "avg_env_reward_last_50", "Env Reward (last 50)", "Reward")
    _plot_series(axes[0, 1], rows, "avg_p_time_percent", "P Time (%)", "Percent")
    _plot_series(axes[1, 0], rows, "avg_c_time_percent", "C Time (%)", "Percent")
    _plot_series(axes[1, 1], rows, "avg_conflict_percent", "Conflict Time (%)", "Percent")

    fig.suptitle(f"Fixed baseline scaling: {method_name}", fontsize=14)
    fig.savefig(args.out_png, dpi=150)
    print(f"Saved fixed scaling plot: {args.out_png}")


if __name__ == "__main__":
    main()
