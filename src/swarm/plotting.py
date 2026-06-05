import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"No rows found in metrics file: {csv_path}")
    keys = rows[0].keys()
    data = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            data[k].append(float(row[k]))
    return {k: np.array(v, dtype=np.float64) for k, v in data.items()}


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    win = min(window, len(x))
    if win <= 1:
        return x.copy()
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, kernel, mode="same")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=str, default="artifacts/training_metrics.csv")
    parser.add_argument("--out_png", type=str, default="artifacts/training_convergence.png")
    parser.add_argument("--ma_window", type=int, default=25, help="Moving average window in episodes")
    return parser


def main():
    args = build_parser().parse_args()
    if args.ma_window <= 0:
        raise ValueError("--ma_window must be positive")

    data = load_metrics(Path(args.metrics_csv))
    episodes = data["episode"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    ax = axes.ravel()

    ax[0].plot(episodes, data["episode_env_reward"], alpha=0.25, label="episode")
    ax[0].plot(episodes, moving_average(data["episode_env_reward"], args.ma_window), label=f"MA({args.ma_window})")
    ax[0].plot(episodes, data["avg_env_reward_last_50"], label="avg_last_50")
    ax0b = ax[0].twinx()
    ax0b.plot(episodes, data["avg_p_time_percent"], color="tab:purple", alpha=0.8, label="P_time_%")
    ax0b.plot(episodes, data["avg_c_time_percent"], color="tab:red", alpha=0.75, label="C_time_%")
    ax0b.plot(episodes, data["avg_conflict_percent"], color="tab:brown", alpha=0.75, label="conflict_%")
    ax0b.plot(episodes, data["conflict_instance_count"], color="tab:olive", alpha=0.75, label="conflict_instances")
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax0b.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, loc="best")
    ax[0].set_title("Environment Reward")

    ax[1].plot(episodes, data["epsilon"], label="epsilon")
    ax[1].legend()
    ax[1].set_title("Exploration Schedule")

    ax[2].plot(episodes, data["policy_loss_mean"], label="policy_loss_mean")
    ax[2].plot(episodes, data["critic_loss_mean"], label="critic_loss_mean")
    ax[2].legend()
    ax[2].set_title("Learned Function Losses")

    ax[3].plot(episodes, data["dr_loss_mean"], label="dr_loss_mean")
    ax[3].plot(episodes, data["dr_mixed_reward_mean"], label="dr_mixed_reward_mean")
    ax[3].plot(episodes, data["my_mean_mean"], label="my_mean_mean")
    ax[3].legend()
    ax[3].set_title("DR Model Convergence")

    ax[4].plot(episodes, data["u_solver_mean"], label="u_solver")
    ax[4].plot(episodes, data["u_causer_mean"], label="u_causer")
    ax[4].legend()
    ax[4].set_title("Self-Role Gate Distribution u(a|s)")

    ax[5].plot(episodes, data["v_c_mean"], label="v_C")
    ax[5].plot(episodes, data["v_p_mean"], label="v_P")
    ax[5].legend()
    ax[5].set_title("Others-Model Gate Distribution v(b|s)")

    fig.suptitle("Training Convergence Dashboard", fontsize=14)
    fig.savefig(args.out_png, dpi=150)
    print(f"Saved plot: {args.out_png}")


if __name__ == "__main__":
    main()

