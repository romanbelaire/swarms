"""
Rank-correlation plots: Spearman rho between arm rankings across my_role variants.
One figure per bandit credit trial (per_arm_credited, arm_relative_my_role, step_level).
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm.config import BANDIT_CONFLICT_ARMS_CSV

MY_ROLES = ("solver", "neutral", "causer")
OTHERS_ROLES = ("allc", "allp", "allsame")
MY_ROLE_PAIRS = (
    ("solver", "neutral", "solver vs neutral"),
    ("solver", "causer", "solver vs causer"),
    ("neutral", "causer", "neutral vs causer"),
)


def _load_summary(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"No rows in {csv_path}")
    return rows


def _fixed_arm_name(row: dict[str, str]) -> str:
    if "arm" in row and row["arm"] != "":
        return row["arm"]
    method = row["method"]
    prefix = "fixed_once_"
    if not method.startswith(prefix):
        raise ValueError(f"Cannot parse fixed baseline arm from method={method}")
    return method[len(prefix) :]


def _arm_value_ranks(row: dict[str, str], arms: tuple[str, ...]) -> np.ndarray:
    values = np.array([float(row[f"arm_{arm}_value"]) for arm in arms], dtype=np.float64)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(arms), dtype=np.float64)
    ranks[order] = np.arange(1, len(arms) + 1, dtype=np.float64)
    return ranks


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        raise ValueError("rank vectors must have equal length")
    if len(a) < 2:
        return float("nan")
    # scipy-free Spearman: Pearson on ranks
    a_dev = a - a.mean()
    b_dev = b - b.mean()
    denom = float(np.sqrt((a_dev ** 2).sum() * (b_dev ** 2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((a_dev * b_dev).sum() / denom)


def _collect_rhos(
    rows: list[dict[str, str]],
    arms: tuple[str, ...],
    *,
    n_agents: int,
    others_role: str,
    seed: int,
) -> dict[str, float]:
    by_my_role: dict[str, np.ndarray] = {}
    for row in rows:
        if int(row["n_agents"]) != n_agents:
            continue
        if int(row["seed"]) != seed:
            continue
        if row["others_role"] != others_role:
            continue
        by_my_role[row["my_role"]] = _arm_value_ranks(row, arms)
    rhos: dict[str, float] = {}
    for left, right, label in MY_ROLE_PAIRS:
        if left not in by_my_role or right not in by_my_role:
            raise ValueError(
                f"Missing my_role rows for n_agents={n_agents} seed={seed} others={others_role}: "
                f"have {sorted(by_my_role.keys())}"
            )
        rhos[label] = _spearman_rho(by_my_role[left], by_my_role[right])
    return rhos


def _aggregate_curve(
    rows: list[dict[str, str]],
    arms: tuple[str, ...],
    *,
    others_role: str,
    agent_counts: list[int],
    seeds: list[int],
) -> tuple[list[int], dict[str, list[float]], dict[str, list[float]]]:
    xs = sorted(agent_counts)
    means: dict[str, list[float]] = {label: [] for _, _, label in MY_ROLE_PAIRS}
    stds: dict[str, list[float]] = {label: [] for _, _, label in MY_ROLE_PAIRS}
    for n_agents in xs:
        per_label: dict[str, list[float]] = {label: [] for _, _, label in MY_ROLE_PAIRS}
        for seed in seeds:
            rhos = _collect_rhos(rows, arms, n_agents=n_agents, others_role=others_role, seed=seed)
            for label, rho in rhos.items():
                per_label[label].append(rho)
        for label in means:
            vals = np.array(per_label[label], dtype=np.float64)
            means[label].append(float(np.nanmean(vals)))
            stds[label].append(float(np.nanstd(vals)))
    return xs, means, stds


def plot_credit_mode_trial(
    *,
    summary_csv: Path,
    fixed_csv: Path | None,
    out_png: Path,
    credit_mode: str,
    arms: tuple[str, ...],
):
    rows = _load_summary(summary_csv)
    agent_counts = sorted({int(row["n_agents"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, others_role in zip(axes, OTHERS_ROLES):
        xs, means, stds = _aggregate_curve(rows, arms, others_role=others_role, agent_counts=agent_counts, seeds=seeds)
        x_arr = np.array(xs, dtype=np.float64)
        for _, _, label in MY_ROLE_PAIRS:
            mean_arr = np.array(means[label], dtype=np.float64)
            std_arr = np.array(stds[label], dtype=np.float64)
            ax.plot(x_arr, mean_arr, marker="o", label=label)
            ax.fill_between(x_arr, mean_arr - std_arr, mean_arr + std_arr, alpha=0.15)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"others={others_role}")
        ax.set_xlabel("Number of agents")
        ax.set_ylim(-1.05, 1.05)
        if others_role == OTHERS_ROLES[0]:
            ax.set_ylabel("Spearman rho (arm value ranks)")
        if others_role == OTHERS_ROLES[-1]:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"my_role rank correlation — credit={credit_mode}\n"
        f"(arms ranked by mean UCB value; bands = std over seeds)",
        fontsize=12,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")

    if fixed_csv is not None and fixed_csv.exists():
        fixed_rows = _load_summary(fixed_csv)
        _plot_fixed_alignment(rows, fixed_rows, arms, credit_mode, out_png.with_name(out_png.stem + "_vs_fixed.png"), agent_counts, seeds)


def _fixed_env_ranks(fixed_rows: list[dict[str, str]], arms: tuple[str, ...], n_agents: int) -> np.ndarray:
    rewards = []
    for arm in arms:
        matched = [
            float(row["avg_env_reward_last_50"])
            for row in fixed_rows
            if int(row["n_agents"]) == n_agents and _fixed_arm_name(row) == arm
        ]
        if len(matched) != 1:
            raise ValueError(f"Expected one fixed row for arm={arm} n_agents={n_agents}, got {len(matched)}")
        rewards.append(matched[0])
    values = np.array(rewards, dtype=np.float64)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(arms), dtype=np.float64)
    ranks[order] = np.arange(1, len(arms) + 1, dtype=np.float64)
    return ranks


def _plot_fixed_alignment(
    bandit_rows: list[dict[str, str]],
    fixed_rows: list[dict[str, str]],
    arms: tuple[str, ...],
    credit_mode: str,
    out_png: Path,
    agent_counts: list[int],
    seeds: list[int],
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, others_role in zip(axes, OTHERS_ROLES):
        xs = agent_counts
        for my_role in MY_ROLES:
            rhos = []
            for n_agents in xs:
                seed_rhos = []
                fixed_ranks = _fixed_env_ranks(fixed_rows, arms, n_agents)
                for seed in seeds:
                    bandit_row = [
                        row
                        for row in bandit_rows
                        if int(row["n_agents"]) == n_agents
                        and int(row["seed"]) == seed
                        and row["others_role"] == others_role
                        and row["my_role"] == my_role
                    ]
                    if len(bandit_row) != 1:
                        raise ValueError(f"Expected one bandit row, got {len(bandit_row)}")
                    bandit_ranks = _arm_value_ranks(bandit_row[0], arms)
                    seed_rhos.append(_spearman_rho(fixed_ranks, bandit_ranks))
                rhos.append(float(np.mean(seed_rhos)))
            ax.plot(xs, rhos, marker="o", label=my_role)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"others={others_role}")
        ax.set_xlabel("Number of agents")
        ax.set_ylim(-1.05, 1.05)
        if others_role == OTHERS_ROLES[0]:
            ax.set_ylabel("Spearman rho vs fixed env-reward ranks")
        if others_role == OTHERS_ROLES[-1]:
            ax.legend(fontsize=8)
    fig.suptitle(f"Bandit arm-value ranks vs fixed baseline env reward — credit={credit_mode}", fontsize=12)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_all_credit_modes(
    out_root: Path,
    credit_modes: list[str],
    arms: tuple[str, ...],
    fixed_summary_csv: Path,
):
    fixed_csv = fixed_summary_csv
    if not fixed_csv.exists():
        raise FileNotFoundError(
            f"Fixed baseline summary not found: {fixed_csv}. "
            "Run run_ablations.py once or pass --fixed_summary_csv."
        )
    for credit_mode in credit_modes:
        trial_dir = out_root / credit_mode
        summary_csv = trial_dir / "my_role_summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing summary: {summary_csv}")
        plots_dir = trial_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_credit_mode_trial(
            summary_csv=summary_csv,
            fixed_csv=fixed_csv,
            out_png=plots_dir / "my_role_rank_correlation.png",
            credit_mode=credit_mode,
            arms=arms,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="artifacts/my_role_credit_ablations")
    parser.add_argument(
        "--fixed_summary_csv",
        type=str,
        default="artifacts/ablations/fixed_conflict_baselines_once.csv",
    )
    parser.add_argument(
        "--credit_modes",
        type=str,
        default="per_arm_credited,arm_relative_my_role,step_level",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default=BANDIT_CONFLICT_ARMS_CSV,
    )
    return parser


def main():
    args = build_parser().parse_args()
    credit_modes = [v.strip() for v in args.credit_modes.split(",") if v.strip()]
    arms = tuple(v.strip() for v in args.arms.split(",") if v.strip())
    plot_all_credit_modes(Path(args.out_root), credit_modes, arms, Path(args.fixed_summary_csv))


if __name__ == "__main__":
    main()
