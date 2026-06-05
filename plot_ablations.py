"""
Plot ablation results: scaling curves and bandit diagnostic figures.
"""

import argparse
import csv
from pathlib import Path
from typing import TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm.config import BANDIT_CONFLICT_ARMS_CSV
from swarm.training.train import (
    BANDIT_ALGORITHM_NAMES,
    BANDIT_REWARD_MODEL_NAMES,
    conflict_grid_artifact_path,
    conflict_local_density_artifact_path,
)

MetricValue: TypeAlias = str | int | float
MetricRow: TypeAlias = dict[str, MetricValue]

ARM_COLORS = {
    "randomwalk3": "#1f77b4",
    "freeze_tag": "#ff7f0e",
    "wait3": "#2ca02c",
    "move_clear": "#d62728",
    "backwards2": "#9467bd",
}

BANDIT_DIAGNOSTIC_SUFFIXES = (
    "prob",
    "value",
    "pull_count",
    "ep_sel_frac",
    "cum_sel_frac",
    "reward_mean",
    "reward_var",
    "delta_mean",
    "delta_var",
    "own_p_mean",
    "local_util_mean",
    "local_util_var",
)

SCALING_METRIC_COLUMNS = (
    "episode_env_reward",
    "avg_env_reward_last_50",
    "avg_p_time_percent",
    "avg_c_time_percent",
    "avg_conflict_percent",
)

CONFLICT_SUMMARY_COLUMNS = (
    "avg_conflict_percent",
    "conflict_instance_count",
    "mean_conflict_instances_per_agent",
)

CONFLICT_METRIC_SCALING_SPECS = (
    (
        "avg_conflict_percent",
        "Conflict steps (%)",
        "Conflict agent-step share at convergence (last episode)",
        "conflict_step_percent_scaling.png",
    ),
    (
        "mean_conflict_instances_per_agent",
        "Mean closed instances / agent",
        "Closed conflict instances per agent at convergence (last episode)",
        "conflict_instances_per_agent_scaling.png",
    ),
)


def _load_summary(csv_path: Path) -> list[MetricRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"No rows in summary file: {csv_path}")
    return cast(list[MetricRow], rows)


def _load_last_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"Metrics CSV has no rows: {csv_path}")
    return rows[-1]


def _load_run_last_rows(rows: list[MetricRow], *, metrics_dir: Path) -> dict[str, dict[str, str]]:
    run_last_rows: dict[str, dict[str, str]] = {}
    run_names = sorted({str(row["run_name"]) for row in rows})
    for run_name in run_names:
        run_csv = metrics_dir / f"{run_name}.csv"
        if not run_csv.exists():
            raise ValueError(f"Missing per-run metrics CSV for convergence heatmap: {run_csv}")
        run_last_rows[run_name] = _load_last_row(run_csv)
    return run_last_rows


def _methods_in_order(rows: list[MetricRow]) -> list[str]:
    order_pairs = sorted({(int(row["plot_order"]), str(row["method"])) for row in rows})
    methods = [method for _, method in order_pairs]
    expected = len(BANDIT_REWARD_MODEL_NAMES)
    if len(methods) != expected:
        raise ValueError(f"Expected exactly {expected} DR methods, got {len(methods)}: {methods}")
    return methods


def _group_stats(rows: list[MetricRow], method: str, metric: str):
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
    from swarm.training.train import baseline_bandit_arms_tuple

    return baseline_bandit_arms_tuple(raw)


def _bandit_diagnostic_column_names(arm_names: tuple[str, ...]) -> list[str]:
    return [f"arm_{arm_name}_{suffix}" for arm_name in arm_names for suffix in BANDIT_DIAGNOSTIC_SUFFIXES]


def _metrics_dir_from_args(args: argparse.Namespace) -> Path:
    if args.metrics_dir != "":
        return Path(args.metrics_dir)
    return Path(args.summary_csv).parent


def _parse_discovered_run(
    csv_path: Path,
    *,
    bandit_algorithm: str,
    reward_model: str,
    run_suffix: str,
) -> tuple[int, int]:
    prefix = f"dr_{bandit_algorithm}_{reward_model}{run_suffix}_agents"
    suffix = csv_path.stem.removeprefix(prefix)
    n_agents_raw, seed_raw = suffix.split("_seed")
    return int(n_agents_raw), int(seed_raw)


def _summary_row_from_metrics(
    *,
    csv_path: Path,
    bandit_algorithm: str,
    reward_model: str,
    plot_order: int,
    n_agents: int,
    seed: int,
    last: dict[str, str],
) -> MetricRow:
    method = f"dr_{bandit_algorithm}_{reward_model}"
    row: MetricRow = {
        "run_name": csv_path.stem,
        "method": method,
        "plot_order": plot_order,
        "n_agents": n_agents,
        "seed": seed,
    }
    for column in SCALING_METRIC_COLUMNS:
        row[column] = float(last[column])
    for column in CONFLICT_SUMMARY_COLUMNS:
        row[column] = float(last[column])
    return row


def _diagnostics_row_from_metrics(
    *,
    csv_path: Path,
    bandit_algorithm: str,
    reward_model: str,
    plot_order: int,
    n_agents: int,
    seed: int,
    last: dict[str, str],
    arm_names: tuple[str, ...],
) -> MetricRow:
    method = f"dr_{bandit_algorithm}_{reward_model}"
    row: MetricRow = {
        "run_name": csv_path.stem,
        "method": method,
        "plot_order": plot_order,
        "n_agents": n_agents,
        "seed": seed,
    }
    for column in _bandit_diagnostic_column_names(arm_names):
        row[column] = float(last[column])
    return row


def _load_algorithm_rows_from_metrics(
    args: argparse.Namespace,
) -> tuple[list[MetricRow], list[MetricRow]]:
    metrics_dir = _metrics_dir_from_args(args)
    arm_names = _parse_bandit_arm_names(args.bandit_conflict_arms)
    summary_rows: list[MetricRow] = []
    diagnostics_rows: list[MetricRow] = []

    run_suffix = args.run_suffix
    for plot_order, reward_model in enumerate(BANDIT_REWARD_MODEL_NAMES):
        glob_pattern = f"dr_{args.bandit_algorithm}_{reward_model}{run_suffix}_agents*_seed*.csv"
        for csv_path in sorted(metrics_dir.glob(glob_pattern)):
            n_agents, seed = _parse_discovered_run(
                csv_path,
                bandit_algorithm=args.bandit_algorithm,
                reward_model=reward_model,
                run_suffix=run_suffix,
            )
            last = _load_last_row(csv_path)
            summary_rows.append(
                _summary_row_from_metrics(
                    csv_path=csv_path,
                    bandit_algorithm=args.bandit_algorithm,
                    reward_model=reward_model,
                    plot_order=plot_order,
                    n_agents=n_agents,
                    seed=seed,
                    last=last,
                )
            )
            diagnostics_rows.append(
                _diagnostics_row_from_metrics(
                    csv_path=csv_path,
                    bandit_algorithm=args.bandit_algorithm,
                    reward_model=reward_model,
                    plot_order=plot_order,
                    n_agents=n_agents,
                    seed=seed,
                    last=last,
                    arm_names=arm_names,
                )
            )

    if len(summary_rows) == 0:
        raise ValueError(f"No per-run CSVs found for bandit_algorithm={args.bandit_algorithm!r} in {metrics_dir}")

    row_order = lambda row: (int(row["n_agents"]), int(row["seed"]), int(row["plot_order"]))
    summary_rows.sort(key=row_order)
    diagnostics_rows.sort(key=row_order)
    return summary_rows, diagnostics_rows


def _aggregate_metric(
    rows: list[MetricRow],
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
    rows: list[MetricRow],
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


def _plot_metric_scaling(
    rows: list[MetricRow],
    fixed_rows: list[MetricRow],
    *,
    metric: str,
    ylabel: str,
    suptitle: str,
    out_png: Path,
) -> None:
    methods = _methods_in_order(rows)
    fixed_methods = sorted({row["method"] for row in fixed_rows})
    fixed_cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for idx, method in enumerate(methods):
        ax = axes.ravel()[idx]
        xs, means, stds = _group_stats(rows, method, metric)
        if len(xs) == 0:
            raise ValueError(f"No data found for method={method}, metric={metric}")
        x_arr = np.array(xs, dtype=np.float64)
        mean_arr = np.array(means, dtype=np.float64)
        std_arr = np.array(stds, dtype=np.float64)
        ax.plot(x_arr, mean_arr, marker="o", color="tab:blue")
        ax.fill_between(x_arr, mean_arr - std_arr, mean_arr + std_arr, alpha=0.2, color="tab:blue")
        for fixed_idx, fixed_method in enumerate(fixed_methods):
            fixed_xs, fixed_means, _ = _group_stats(fixed_rows, fixed_method, metric)
            if len(fixed_xs) == 0:
                raise ValueError(f"No data found for fixed baseline method={fixed_method}, metric={metric}")
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
        ax.set_ylabel(ylabel)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(out_png, dpi=150)
    print(f"Saved ablation plot: {out_png}")


def _plot_scaling(
    rows: list[MetricRow],
    fixed_rows: list[MetricRow],
    *,
    out_png: Path,
) -> None:
    _plot_metric_scaling(
        rows,
        fixed_rows,
        metric="avg_env_reward_last_50",
        ylabel="Mean Reward (last 50)",
        suptitle="DR Reward Bandit Ablation with Fixed Baselines: Mean Reward vs Number of Agents",
        out_png=out_png,
    )


def _plot_conflict_metric_scaling(
    rows: list[MetricRow],
    fixed_rows: list[MetricRow],
    *,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, suptitle, filename in CONFLICT_METRIC_SCALING_SPECS:
        _plot_metric_scaling(
            rows,
            fixed_rows,
            metric=metric,
            ylabel=ylabel,
            suptitle=suptitle,
            out_png=out_dir / filename,
        )


def _plot_arm_selection(
    rows: list[MetricRow],
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
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


def _arm_metric_mean_std(
    rows: list[MetricRow],
    *,
    method: str,
    n_agents: int,
    arm: str,
    column_suffix: str,
) -> tuple[float, float]:
    if column_suffix == "delta_mean":
        return _aggregate_delta(rows, method=method, n_agents=n_agents, arm=arm)
    return _aggregate_metric(
        rows,
        method=method,
        n_agents=n_agents,
        column=f"arm_{arm}_{column_suffix}",
    )


def _plot_per_arm_metric_grid(
    rows: list[MetricRow],
    *,
    methods: list[str],
    arms: tuple[str, ...],
    agent_counts: list[int],
    column_suffix: str,
    ylabel: str,
    suptitle: str,
    out_png: Path,
    y_reference: float | None = None,
) -> None:
    n_arms = len(arms)
    n_ns = len(agent_counts)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
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
                mean_val, std_val = _arm_metric_mean_std(
                    rows,
                    method=method,
                    n_agents=n_agents,
                    arm=arm,
                    column_suffix=column_suffix,
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
        ax.set_ylabel(ylabel)
        if y_reference is not None:
            ax.axhline(y_reference, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_arms, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.suptitle(suptitle, fontsize=12, y=1.04)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved bandit metric plot: {out_png}")


BANDIT_METRIC_GRID_SPECS = (
    (
        "delta_mean",
        "mean DR delta D",
        "Per-arm mean DR delta D by scenario (instance-level, pre-softplus)",
        "bandit_arm_delta_grid.png",
        0.0,
    ),
    (
        "value",
        "UCB Q estimate",
        "Per-arm UCB Q estimate by DR scenario at convergence",
        "bandit_arm_q_grid.png",
        None,
    ),
    (
        "pull_count",
        "mean pull count",
        "Per-arm UCB pull count by DR scenario at convergence",
        "bandit_arm_pulls_grid.png",
        None,
    ),
    (
        "own_p_mean",
        "own P share",
        "Per-arm own P share by DR scenario (length-normalized instance)",
        "bandit_arm_own_p_grid.png",
        None,
    ),
)


def _plot_bandit_metric_grids(
    rows: list[MetricRow],
    *,
    methods: list[str],
    arms: tuple[str, ...],
    agent_counts: list[int],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for column_suffix, ylabel, suptitle, filename, y_reference in BANDIT_METRIC_GRID_SPECS:
        _plot_per_arm_metric_grid(
            rows,
            methods=methods,
            arms=arms,
            agent_counts=agent_counts,
            column_suffix=column_suffix,
            ylabel=ylabel,
            suptitle=suptitle,
            out_png=out_dir / filename,
            y_reference=y_reference,
        )


def _plot_local_util_grid(
    rows: list[MetricRow],
    *,
    methods: list[str],
    arms: tuple[str, ...],
    agent_counts: list[int],
    out_png: Path,
) -> None:
    _plot_per_arm_metric_grid(
        rows,
        methods=methods,
        arms=arms,
        agent_counts=agent_counts,
        column_suffix="local_util_mean",
        ylabel="agent-mean local util",
        suptitle="Per-arm agent-mean local util by DR (length-normalized observed_util input)",
        out_png=out_png,
        y_reference=0.0,
    )


def _plot_bandit_diagnostics_rows(
    rows: list[MetricRow],
    *,
    args: argparse.Namespace,
) -> None:
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
    _plot_bandit_metric_grids(
        rows,
        methods=methods,
        arms=arms,
        agent_counts=agent_counts,
        out_dir=Path(args.out_bandit_metrics_dir),
    )
    _plot_local_util_grid(
        rows,
        methods=methods,
        arms=arms,
        agent_counts=agent_counts,
        out_png=Path(args.out_local_util_grid),
    )
    _plot_swarm_behavioral_convergence(
        rows,
        methods=methods,
        arms=arms,
        agent_counts=agent_counts,
        metrics_dir=_metrics_dir_from_args(args),
        out_png=Path(args.out_swarm_convergence),
    )
    _plot_conflict_location_heatmap(
        rows,
        methods=methods,
        agent_counts=agent_counts,
        metrics_dir=_metrics_dir_from_args(args),
        out_png=Path(args.out_conflict_heatmap),
    )
    _plot_conflict_local_density_heatmap(
        rows,
        methods=methods,
        agent_counts=agent_counts,
        metrics_dir=_metrics_dir_from_args(args),
        out_png=Path(args.out_conflict_local_heatmap),
    )
    if args.conflict_local_n30_arm_split in agent_counts:
        _plot_conflict_local_density_n30_by_arm(
            rows,
            methods=methods,
            arms=arms,
            n_agents=args.conflict_local_n30_arm_split,
            metrics_dir=_metrics_dir_from_args(args),
            out_png=Path(args.out_conflict_local_n30_by_arm),
        )


def _swarm_convergence_matrix_for_method_n(
    rows: list[MetricRow],
    *,
    run_last_rows: dict[str, dict[str, str]],
    method: str,
    n_agents: int,
    arms: tuple[str, ...],
) -> np.ndarray:
    candidate_rows = [row for row in rows if row["method"] == method and int(row["n_agents"]) == n_agents]
    if len(candidate_rows) == 0:
        raise ValueError(f"No diagnostic rows for method={method}, n_agents={n_agents}")
    chosen = min(candidate_rows, key=lambda row: int(row["seed"]))
    last = run_last_rows[str(chosen["run_name"])]
    matrix = np.zeros((n_agents, len(arms)), dtype=np.float64)
    for agent_idx in range(n_agents):
        for arm_idx, arm in enumerate(arms):
            col = f"arm_{arm}_agent_{agent_idx}_prob"
            if col not in last:
                raise ValueError(
                    f"Missing per-agent probability column {col!r} in run {chosen['run_name']}. "
                    "Re-run ablations to regenerate metrics with per-agent arm probabilities."
                )
            matrix[agent_idx, arm_idx] = float(last[col])
    return matrix


def _load_conflict_grid_counts(metrics_dir: Path, run_name: str) -> np.ndarray:
    grid_path = conflict_grid_artifact_path(metrics_dir / f"{run_name}.csv")
    if not grid_path.exists():
        raise ValueError(
            f"Missing conflict grid artifact for run {run_name!r}: {grid_path}. "
            "Re-run ablations (without --resume, or delete incomplete runs) to generate conflict grids."
        )
    data = np.load(grid_path)
    return np.asarray(data["counts"], dtype=np.float64)


def _aggregate_conflict_grid_for_method_n(
    rows: list[MetricRow],
    *,
    metrics_dir: Path,
    method: str,
    n_agents: int,
) -> np.ndarray:
    run_rows = [row for row in rows if row["method"] == method and int(row["n_agents"]) == n_agents]
    if len(run_rows) == 0:
        raise ValueError(f"No rows for method={method}, n_agents={n_agents}")
    aggregated = None
    for row in run_rows:
        counts = _load_conflict_grid_counts(metrics_dir, str(row["run_name"]))
        if aggregated is None:
            aggregated = counts.copy()
            continue
        if aggregated.shape != counts.shape:
            raise ValueError(
                f"Conflict grid shape mismatch for method={method}, n_agents={n_agents}: "
                f"{aggregated.shape} vs {counts.shape}"
            )
        aggregated += counts
    total = float(aggregated.sum())
    if total <= 0.0:
        raise ValueError(f"Conflict grid is empty for method={method}, n_agents={n_agents}")
    return aggregated / total


def _load_conflict_local_density_counts(metrics_dir: Path, run_name: str) -> np.ndarray:
    grid_path = conflict_local_density_artifact_path(metrics_dir / f"{run_name}.csv")
    if not grid_path.exists():
        raise ValueError(
            f"Missing conflict local density artifact for run {run_name!r}: {grid_path}. "
            "Re-run ablations (without --resume, or delete incomplete runs) to generate density grids."
        )
    data = np.load(grid_path)
    return np.asarray(data["counts"], dtype=np.float64)


def _load_conflict_local_density_arm_counts(metrics_dir: Path, run_name: str, arm: str) -> np.ndarray:
    grid_path = conflict_local_density_artifact_path(metrics_dir / f"{run_name}.csv")
    if not grid_path.exists():
        raise ValueError(f"Missing conflict local density artifact for run {run_name!r}: {grid_path}")
    data = np.load(grid_path)
    key = f"arm_{arm}_counts"
    if key not in data:
        raise ValueError(
            f"Missing per-arm conflict local density {key!r} in {grid_path}. "
            "Re-run bandit_ucb1 ablations to regenerate arm-split density grids."
        )
    return np.asarray(data[key], dtype=np.float64)


def _aggregate_conflict_local_density_for_method_n(
    rows: list[MetricRow],
    *,
    metrics_dir: Path,
    method: str,
    n_agents: int,
) -> np.ndarray:
    run_rows = [row for row in rows if row["method"] == method and int(row["n_agents"]) == n_agents]
    if len(run_rows) == 0:
        raise ValueError(f"No rows for method={method}, n_agents={n_agents}")
    aggregated = None
    for row in run_rows:
        counts = _load_conflict_local_density_counts(metrics_dir, str(row["run_name"]))
        if aggregated is None:
            aggregated = counts.copy()
            continue
        if aggregated.shape != counts.shape:
            raise ValueError(
                f"Conflict local density shape mismatch for method={method}, n_agents={n_agents}: "
                f"{aggregated.shape} vs {counts.shape}"
            )
        aggregated += counts
    total = float(aggregated.sum())
    if total <= 0.0:
        raise ValueError(f"Conflict local density grid is empty for method={method}, n_agents={n_agents}")
    return aggregated / total


def _aggregate_conflict_local_density_arm_for_method_n(
    rows: list[MetricRow],
    *,
    metrics_dir: Path,
    method: str,
    n_agents: int,
    arm: str,
) -> np.ndarray:
    run_rows = [row for row in rows if row["method"] == method and int(row["n_agents"]) == n_agents]
    if len(run_rows) == 0:
        raise ValueError(f"No rows for method={method}, n_agents={n_agents}")
    aggregated = None
    for row in run_rows:
        counts = _load_conflict_local_density_arm_counts(metrics_dir, str(row["run_name"]), arm)
        if aggregated is None:
            aggregated = counts.copy()
            continue
        if aggregated.shape != counts.shape:
            raise ValueError(
                f"Arm conflict local density shape mismatch for method={method}, n_agents={n_agents}, arm={arm}"
            )
        aggregated += counts
    total = float(aggregated.sum())
    if total <= 0.0:
        raise ValueError(
            f"Arm conflict local density grid is empty for method={method}, n_agents={n_agents}, arm={arm}"
        )
    return aggregated / total


def _plot_conflict_local_heatmap_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    title: str,
    ylabel: str,
    show_ylabel: bool,
    cmap: str,
):
    image = ax.imshow(matrix, origin="lower", cmap=cmap, aspect="equal", vmin=0.0)
    local_size = matrix.shape[0]
    ax.set_xticks(np.arange(local_size))
    ax.set_yticks(np.arange(local_size))
    ax.set_xlabel("local j", fontsize=8)
    ax.set_title(title, fontsize=9)
    if show_ylabel:
        ax.set_ylabel(f"{ylabel}\nlocal i", fontsize=8)
    else:
        ax.set_ylabel("")
    return image


def _plot_conflict_local_density_heatmap(
    rows: list[MetricRow],
    *,
    methods: list[str],
    agent_counts: list[int],
    metrics_dir: Path,
    out_png: Path,
) -> None:
    n_rows = len(methods)
    n_cols = len(agent_counts)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols + 2.0, 2.6 * n_rows + 1.2),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for method_idx, method in enumerate(methods):
        for n_idx, n_agents in enumerate(agent_counts):
            ax = axes[method_idx, n_idx]
            matrix = _aggregate_conflict_local_density_for_method_n(
                rows,
                metrics_dir=metrics_dir,
                method=method,
                n_agents=n_agents,
            )
            image = _plot_conflict_local_heatmap_panel(
                ax,
                matrix,
                title=f"N={n_agents}",
                ylabel=method,
                show_ylabel=n_idx == 0,
                cmap="plasma",
            )
    if image is None:
        raise ValueError("Failed to create conflict local density heatmap")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, label="Masked peer tile share")
    fig.suptitle(
        "Conflict local density: other-agent tiles (3/4) in egocentric 5×5 obs (summed over seeds)",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved conflict local density heatmap: {out_png}")


def _plot_conflict_local_density_n30_by_arm(
    rows: list[MetricRow],
    *,
    methods: list[str],
    arms: tuple[str, ...],
    n_agents: int,
    metrics_dir: Path,
    out_png: Path,
) -> None:
    n_rows = len(methods)
    n_cols = len(arms)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols + 2.0, 2.6 * n_rows + 1.2),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for method_idx, method in enumerate(methods):
        for arm_idx, arm in enumerate(arms):
            ax = axes[method_idx, arm_idx]
            matrix = _aggregate_conflict_local_density_arm_for_method_n(
                rows,
                metrics_dir=metrics_dir,
                method=method,
                n_agents=n_agents,
                arm=arm,
            )
            image = _plot_conflict_local_heatmap_panel(
                ax,
                matrix,
                title=arm,
                ylabel=method,
                show_ylabel=arm_idx == 0,
                cmap="plasma",
            )
    if image is None:
        raise ValueError("Failed to create N=30 arm-split conflict local density heatmap")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, label="Arm-conditioned peer tile share")
    fig.suptitle(
        f"Conflict local density at N={n_agents} by active bandit arm (egocentric 5×5, tiles 3/4 only)",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved conflict local density N={n_agents} by-arm heatmap: {out_png}")


def _plot_conflict_location_heatmap(
    rows: list[MetricRow],
    *,
    methods: list[str],
    agent_counts: list[int],
    metrics_dir: Path,
    out_png: Path,
) -> None:
    n_rows = len(methods)
    n_cols = len(agent_counts)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols + 2.0, 2.6 * n_rows + 1.2),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for method_idx, method in enumerate(methods):
        for n_idx, n_agents in enumerate(agent_counts):
            ax = axes[method_idx, n_idx]
            matrix = _aggregate_conflict_grid_for_method_n(
                rows,
                metrics_dir=metrics_dir,
                method=method,
                n_agents=n_agents,
            )
            image = ax.imshow(matrix.T, origin="lower", cmap="hot", aspect="equal", vmin=0.0)
            grid_size = matrix.shape[0]
            ax.set_xticks(np.arange(grid_size))
            ax.set_yticks(np.arange(grid_size))
            ax.set_xlabel("x", fontsize=8)
            if method_idx == 0:
                ax.set_title(f"N={n_agents}", fontsize=9)
            if n_idx == 0:
                ax.set_ylabel(f"{method}\ny", fontsize=8)
            else:
                ax.set_ylabel("")
    if image is None:
        raise ValueError("Failed to create conflict location heatmap")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, label="Conflict occupancy share")
    fig.suptitle(
        "Conflict locations on state grid (share of in-conflict agent-steps; summed over seeds)",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved conflict location heatmap: {out_png}")


def _plot_swarm_behavioral_convergence(
    rows: list[MetricRow],
    *,
    methods: list[str],
    arms: tuple[str, ...],
    agent_counts: list[int],
    metrics_dir: Path,
    out_png: Path,
) -> None:
    run_last_rows = _load_run_last_rows(rows, metrics_dir=metrics_dir)
    n_rows = len(methods)
    n_cols = len(agent_counts)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols + 2.0, 2.6 * n_rows + 1.2),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for method_idx, method in enumerate(methods):
        for n_idx, n_agents in enumerate(agent_counts):
            ax = axes[method_idx, n_idx]
            matrix = _swarm_convergence_matrix_for_method_n(
                rows,
                run_last_rows=run_last_rows,
                method=method,
                n_agents=n_agents,
                arms=arms,
            )
            image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_xticks(np.arange(len(arms)))
            ax.set_xticklabels(arms, rotation=45, ha="right", fontsize=7)
            y_ticks = np.arange(n_agents, dtype=np.int32)
            if n_agents > 20:
                stride = int(np.ceil(n_agents / 20))
                y_ticks = y_ticks[::stride]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f"agent_{idx}" for idx in y_ticks], fontsize=6)
            if method_idx == 0:
                ax.set_title(f"N={n_agents}", fontsize=9)
            if n_idx == 0:
                ax.set_ylabel(f"{method}\nagent", fontsize=8)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("arm", fontsize=8)
    if image is None:
        raise ValueError("Failed to create convergence heatmap image")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, label="Pull probability")
    fig.suptitle(
        "Swarm Behavioral Convergence: per-agent arm pull probabilities (homogeneous rows vs specialization)",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved swarm behavioral convergence plot: {out_png}")


def _plot_bandit_diagnostics(args: argparse.Namespace) -> None:
    rows = _load_summary(Path(args.bandit_summary_csv))
    _plot_bandit_diagnostics_rows(rows, args=args)


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
        "--out_conflict_metrics_dir",
        type=str,
        default="",
        help="Directory for conflict metric scaling PNGs (default: parent of --summary_csv).",
    )
    parser.add_argument(
        "--bandit_summary_csv",
        type=str,
        default="artifacts/ablations/bandit_diagnostics_summary.csv",
    )
    parser.add_argument("--out_arm_selection", type=str, default="artifacts/ablations/bandit_arm_selection.png")
    parser.add_argument(
        "--out_bandit_metrics_dir",
        type=str,
        default="artifacts/ablations",
        help="Directory for per-metric bandit grids (delta, Q, pulls, own_p)",
    )
    parser.add_argument("--out_local_util_grid", type=str, default="artifacts/ablations/bandit_local_util_grid.png")
    parser.add_argument(
        "--out_swarm_convergence",
        type=str,
        default="artifacts/ablations/bandit_swarm_behavioral_convergence.png",
    )
    parser.add_argument(
        "--out_conflict_heatmap",
        type=str,
        default="artifacts/ablations/conflict_location_heatmap.png",
    )
    parser.add_argument(
        "--out_conflict_local_heatmap",
        type=str,
        default="artifacts/ablations/conflict_local_density_heatmap.png",
    )
    parser.add_argument(
        "--out_conflict_local_n30_by_arm",
        type=str,
        default="artifacts/ablations/conflict_local_density_heatmap_n30_by_arm.png",
    )
    parser.add_argument(
        "--conflict_local_n30_arm_split",
        type=int,
        default=30,
        help="Agent count for arm-split conflict local density figure (skipped if absent from data).",
    )
    parser.add_argument(
        "--n_agents",
        type=int,
        default=None,
        help="Optional filter for arm-selection plot only (default: all N on x-axis)",
    )
    parser.add_argument(
        "--bandit_conflict_arms",
        type=str,
        default=BANDIT_CONFLICT_ARMS_CSV,
    )
    parser.add_argument(
        "--bandit_algorithm",
        type=str,
        default="",
        choices=["", *BANDIT_ALGORITHM_NAMES],
        help=(
            "When set, ignore the summary CSV inputs for DR runs and reconstruct them by scanning "
            "--metrics_dir for dr_<algorithm>_<reward_model>_agents<N>_seed<S>.csv files."
        ),
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="",
        help="Directory containing per-run CSV files when --bandit_algorithm is set (default: parent of --summary_csv).",
    )
    parser.add_argument(
        "--run_suffix",
        type=str,
        default="",
        help="Run-name token before _agents (e.g. _dualqb for dual_quadrant_base ablations).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.bandit_algorithm == "":
        rows = _load_summary(Path(args.summary_csv))
        diagnostics_rows = _load_summary(Path(args.bandit_summary_csv))
    else:
        rows, diagnostics_rows = _load_algorithm_rows_from_metrics(args)

    fixed_rows = _load_summary(Path(args.fixed_summary_csv))
    _plot_scaling(rows, fixed_rows, out_png=Path(args.out_png))
    conflict_metrics_dir = (
        Path(args.out_conflict_metrics_dir)
        if args.out_conflict_metrics_dir != ""
        else Path(args.summary_csv).parent
    )
    _plot_conflict_metric_scaling(rows, fixed_rows, out_dir=conflict_metrics_dir)
    _plot_bandit_diagnostics_rows(diagnostics_rows, args=args)


if __name__ == "__main__":
    main()
