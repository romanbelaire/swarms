"""
Plot ablation results: scaling curves and bandit diagnostic figures.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARM_COLORS = {
    "randomwalk3": "#1f77b4",
    "freeze_tag": "#ff7f0e",
    "wait3": "#2ca02c",
    "move_clear": "#d62728",
    "backward3": "#9467bd",
    "wait2_forward1": "#8c564b",
    "handshake": "#e377c2",
    "reserve_parity": "#7f7f7f",
    "reserve_parity_escape": "#bcbd22",
    "priority_swap_n3": "#17becf",
    "pass_food_n3": "#aec7e8",
}


def _load_summary(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"No rows in summary file: {csv_path}")
    return rows


def _methods_in_order(rows: list[dict[str, str]]) -> list[str]:
    order_pairs = sorted({(int(row["plot_order"]), row["method"]) for row in rows})
    methods = [method for _, method in order_pairs]
    if len(methods) != 9:
        raise ValueError(f"Expected exactly 9 DR methods, got {len(methods)}")
    return methods


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


def _parse_bandit_arm_names(raw: str) -> tuple[str, ...]:
    arm_names = tuple(name.strip() for name in raw.split(",") if name.strip() != "")
    if len(arm_names) == 0:
        raise ValueError("bandit_conflict_arms must list at least one arm")
    return arm_names


def _aggregate_metric(
    rows: list[dict[str, str]],
    *,
    method: str,
    n_agents: int,
    column: str,
) -> tuple[float, float]:
    values = [float(row[column]) for row in rows if row["method"] == method and int(row["n_agents"]) == n_agents]
    if len(values) == 0:
        raise ValueError(f"No rows for method={method}, n_agents={n_agents}, column={column}")
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def _aggregate_delta(
    rows: list[dict[str, str]],
    *,
    method: str,
    n_agents: int,
    arm: str,
) -> tuple[float, float]:
    mean_val, std_val = _aggregate_metric(
        rows,
        method=method,
        n_agents=n_agents,
        column=f"arm_{arm}_delta_mean",
    )
    return mean_val, std_val


def _plot_scaling(
    rows: list[dict[str, str]],
    fixed_rows: list[dict[str, str]],
    *,
    out_png: Path,
) -> None:
    methods = _methods_in_order(rows)
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
    fig.savefig(out_png, dpi=150)
    print(f"Saved ablation plot: {out_png}")


def _plot_arm_selection(
    rows: list[dict[str, str]],
    *,
    methods: list[str],
    arms: tuple[str, ...],
    agent_counts: list[int],
    n_agents_filter: int | None,
    out_png: Path,
) -> None:
    if n_agents_filter is not None:
        plot_agent_counts = [n_agents_filter]
    else:
        plot_agent_counts = agent_counts

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    x = np.arange(len(plot_agent_counts), dtype=np.float64)
    for idx, method in enumerate(methods):
        ax = axes.ravel()[idx]
        bottom = np.zeros(len(plot_agent_counts), dtype=np.float64)
        for arm in arms:
            means = []
            for n_agents in plot_agent_counts:
                mean_prob, _ = _aggregate_metric(
                    rows, method=method, n_agents=n_agents, column=f"arm_{arm}_prob"
                )
                means.append(mean_prob)
            heights = np.array(means, dtype=np.float64)
            color = ARM_COLORS.get(arm, None)
            ax.bar(x, heights, bottom=bottom, label=arm, color=color)
            bottom = bottom + heights
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in plot_agent_counts], fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel("UCB pull share")
        ax.set_title(method, fontsize=9)
        ax.axhline(1.0 / len(arms), color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(arms), bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.suptitle(
        "DR Bandit Arm Selection at Convergence (stack height = UCB pull probability; uniform ≈ mixing)",
        fontsize=12,
        y=1.04,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved arm selection plot: {out_png}")


def _plot_diagnostics_grid(
    rows: list[dict[str, str]],
    *,
    methods: list[str],
    arms: tuple[str, ...],
    agent_counts: list[int],
    out_png: Path,
) -> None:
    n_arms = len(arms)
    n_ns = len(agent_counts)
    n_metrics = 4
    metric_specs = (
        ("delta_mean", "D"),
        ("value", "Q (norm)"),
        ("pull_count", "pulls (norm)"),
        ("own_p_mean", "own_p"),
    )

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    x_centers = np.arange(n_ns, dtype=np.float64)

    for idx, method in enumerate(methods):
        ax = axes.ravel()[idx]
        group_width = 0.75
        cluster_width = group_width / n_arms
        pull_values: list[float] = []
        for n_agents in agent_counts:
            for arm in arms:
                mean_pull, _ = _aggregate_metric(
                    rows, method=method, n_agents=n_agents, column=f"arm_{arm}_pull_count"
                )
                pull_values.append(mean_pull)
        pull_max = float(np.max(pull_values))
        if pull_max <= 0.0:
            pull_max = 1.0
        q_values: list[float] = []
        for n_agents in agent_counts:
            for arm in arms:
                mean_q, _ = _aggregate_metric(
                    rows, method=method, n_agents=n_agents, column=f"arm_{arm}_value"
                )
                q_values.append(mean_q)
        q_max = float(np.max(q_values))
        if q_max <= 0.0:
            q_max = 1.0

        for arm_idx, arm in enumerate(arms):
            cluster_offset = -group_width / 2.0 + cluster_width / 2.0 + arm_idx * cluster_width
            metric_offsets = np.linspace(-cluster_width * 0.4, cluster_width * 0.4, n_metrics)
            arm_color = ARM_COLORS.get(arm, None)
            for n_idx, n_agents in enumerate(agent_counts):
                base_x = x_centers[n_idx]
                for metric_idx, (mean_col, _label) in enumerate(metric_specs):
                    bar_x = base_x + cluster_offset + metric_offsets[metric_idx]
                    if mean_col == "delta_mean":
                        mean_val, std_val = _aggregate_delta(
                            rows, method=method, n_agents=n_agents, arm=arm
                        )
                        height = mean_val
                        yerr = std_val
                    elif mean_col == "pull_count":
                        mean_val, _ = _aggregate_metric(
                            rows, method=method, n_agents=n_agents, column=f"arm_{arm}_{mean_col}"
                        )
                        height = mean_val / pull_max
                        yerr = 0.0
                    elif mean_col == "value":
                        mean_val, _ = _aggregate_metric(
                            rows, method=method, n_agents=n_agents, column=f"arm_{arm}_{mean_col}"
                        )
                        height = mean_val / q_max
                        yerr = 0.0
                    else:
                        mean_val, std_val = _aggregate_metric(
                            rows, method=method, n_agents=n_agents, column=f"arm_{arm}_{mean_col}"
                        )
                        height = mean_val
                        yerr = std_val
                    hatch = "" if metric_idx < 3 else "///"
                    ax.bar(
                        bar_x,
                        height,
                        width=cluster_width / (n_metrics + 1.0),
                        color=arm_color,
                        alpha=0.55 if metric_idx >= 3 else 0.85,
                        hatch=hatch,
                        yerr=yerr,
                        capsize=2,
                        error_kw={"elinewidth": 0.8},
                        label=arm if idx == 0 and n_idx == 0 and metric_idx == 0 else None,
                    )

        ax.set_title(method, fontsize=9)
        ax.set_xticks(x_centers)
        ax.set_xticklabels([str(n) for n in agent_counts], fontsize=8)
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel("metric value")
        ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ARM_COLORS.get(arm, "gray"), label=arm)
        for arm in arms
    ]
    legend_handles.extend(
        [
            plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.85, label="D / Q / pulls"),
            plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.55, hatch="///", label="own_p"),
        ]
    )
    fig.legend(handles=legend_handles, loc="upper center", ncol=min(n_arms + 2, 8), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Per-arm bandit diagnostics by DR (x = N; D = logged delta; Q & pulls scaled per subplot)",
        fontsize=12,
        y=1.04,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved diagnostics grid plot: {out_png}")


def _plot_team_util_grid(
    rows: list[dict[str, str]],
    *,
    methods: list[str],
    arms: tuple[str, ...],
    agent_counts: list[int],
    out_png: Path,
) -> None:
    n_arms = len(arms)
    n_ns = len(agent_counts)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    x_centers = np.arange(n_ns, dtype=np.float64)

    for idx, method in enumerate(methods):
        ax = axes.ravel()[idx]
        group_width = 0.75
        bar_width = group_width / n_arms
        for arm_idx, arm in enumerate(arms):
            offset = -group_width / 2.0 + bar_width / 2.0 + arm_idx * bar_width
            arm_color = ARM_COLORS.get(arm, None)
            heights: list[float] = []
            yerrs: list[float] = []
            bar_xs: list[float] = []
            for n_idx, n_agents in enumerate(agent_counts):
                mean_val, std_val = _aggregate_metric(
                    rows, method=method, n_agents=n_agents, column=f"arm_{arm}_team_util_mean"
                )
                bar_xs.append(x_centers[n_idx] + offset)
                heights.append(mean_val)
                yerrs.append(std_val)
            ax.bar(
                bar_xs,
                heights,
                width=bar_width * 0.9,
                color=arm_color,
                yerr=yerrs,
                capsize=2,
                error_kw={"elinewidth": 0.8},
                label=arm if idx == 0 else None,
            )

        ax.set_title(method, fontsize=9)
        ax.set_xticks(x_centers)
        ax.set_xticklabels([str(n) for n in agent_counts], fontsize=8)
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel("agent-mean team util")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_arms, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.suptitle(
        "Per-arm agent-mean team util by DR (length-normalized observed_util input)",
        fontsize=12,
        y=1.04,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved team util plot: {out_png}")


def _plot_bandit_diagnostics(args: argparse.Namespace) -> None:
    rows = _load_summary(Path(args.bandit_summary_csv))
    methods = _methods_in_order(rows)
    arms = _parse_bandit_arm_names(args.bandit_conflict_arms)
    agent_counts = sorted({int(row["n_agents"]) for row in rows})

    _plot_arm_selection(
        rows,
        methods=methods,
        arms=arms,
        agent_counts=agent_counts,
        n_agents_filter=args.n_agents,
        out_png=Path(args.out_arm_selection),
    )
    _plot_diagnostics_grid(
        rows,
        methods=methods,
        arms=arms,
        agent_counts=agent_counts,
        out_png=Path(args.out_diagnostics_grid),
    )
    _plot_team_util_grid(
        rows,
        methods=methods,
        arms=arms,
        agent_counts=agent_counts,
        out_png=Path(args.out_team_util_grid),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, default="artifacts/ablations/ablation_summary.csv")
    parser.add_argument(
        "--fixed_summary_csv",
        type=str,
        default="artifacts/ablations/fixed_conflict_baselines_once.csv",
    )
    parser.add_argument("--out_png", type=str, default="artifacts/ablations/ablation_scaling.png")
    parser.add_argument(
        "--bandit_summary_csv",
        type=str,
        default="artifacts/ablations/bandit_diagnostics_summary.csv",
    )
    parser.add_argument("--out_arm_selection", type=str, default="artifacts/ablations/bandit_arm_selection.png")
    parser.add_argument("--out_diagnostics_grid", type=str, default="artifacts/ablations/bandit_arm_diagnostics_grid.png")
    parser.add_argument("--out_team_util_grid", type=str, default="artifacts/ablations/bandit_team_util_grid.png")
    parser.add_argument(
        "--n_agents",
        type=int,
        default=None,
        help="Optional filter for arm-selection plot only (default: all N on x-axis)",
    )
    parser.add_argument(
        "--bandit_conflict_arms",
        type=str,
        default="randomwalk3,freeze_tag,wait3,move_clear,backward3",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    rows = _load_summary(Path(args.summary_csv))
    fixed_rows = _load_summary(Path(args.fixed_summary_csv))
    _plot_scaling(rows, fixed_rows, out_png=Path(args.out_png))
    _plot_bandit_diagnostics(args)


if __name__ == "__main__":
    main()
