"""
Train a full (5-action) DQN on sparse PettingZoo rewards from the foraging env.

Unlike swarm.training.train full mode, policy TD targets use rewards[agent] from env.step
(not critic-shaped DR). infos[*]["env_reward"] is the team total per step (for n_agents=1
it matches the solo agent's reward that step).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ is None:
    _src = Path(__file__).resolve().parents[2]
    _src_s = str(_src)
    if _src_s not in sys.path:
        sys.path.insert(0, _src_s)

from swarm.agents import DEVICE, DQNAgent
from swarm.config import (
    GRID_SIZE,
    EXPERT_DEFAULT_MAX_STEPS_PER_EPISODE,
    EXPERT_EVAL_EVERY_EPISODES,
    NUM_FOOD,
    N_EPISODES,
    OBS_DIM,
    N_ACTIONS_FULL,
)
from swarm.env import RationalSwarmForagingEnv, RationalSwarmForagingVecEnv


def _eval_greedy_mean_team_return(
    env: RationalSwarmForagingEnv,
    policies: dict,
    max_steps: int,
    n_rollouts: int,
    base_seed: int,
) -> float:
    """Mean per-episode sum of infos[*]['env_reward'] (same scale as training logs) under greedy policy."""
    agent_ids = env.possible_agents
    totals: list[float] = []
    for i in range(n_rollouts):
        obs, _ = env.reset(seed=base_seed + 1_000_003 + i)
        ep_ret = 0.0
        for t in range(max_steps):
            actions = {aid: policies[aid].select_action(obs[aid], deterministic=True) for aid in agent_ids}
            obs, _rewards, _terms, _truncs, infos = env.step(actions)
            ep_ret += float(infos[agent_ids[0]]["env_reward"])
        totals.append(ep_ret)
    return float(np.mean(np.array(totals, dtype=np.float64)))


def _best_weight_path(main_out: Path, agent_id: str) -> Path:
    if agent_id == "agent_0":
        return main_out.with_name(main_out.stem + "_best" + main_out.suffix)
    return main_out.parent / f"dqn_weights_{agent_id}_best.pt"


def _save_best_checkpoints(main_out: Path, policies: dict, agent_ids: list[str], n_agents: int) -> None:
    main_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policies["agent_0"].q_net.state_dict(), _best_weight_path(main_out, "agent_0"))
    if n_agents > 1:
        for agent_id in agent_ids:
            if agent_id == "agent_0":
                continue
            torch.save(policies[agent_id].q_net.state_dict(), _best_weight_path(main_out, agent_id))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train expert DQN on raw env rewards (no DR critic shaping).")
    p.add_argument("--n_agents", type=int, default=1, help="Number of agents (default 1 for solo expert).")
    p.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Batched env rows in one step (single-threaded NumPy, not multiprocessed). "
        "Training runs (num_envs * grad_steps) optimizer steps per env step, so large values "
        "usually increase wall time unless you lower --grad_steps.",
    )
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=EXPERT_DEFAULT_MAX_STEPS_PER_EPISODE,
        help="Horizon per episode (solo foraging needs room to reach food and base; DR experiments often use 50).",
    )
    p.add_argument("--alpha", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.995, help="Higher gamma helps sparse delivery credit.")
    p.add_argument("--epsilon_decay", type=int, default=1500, help="Episodes over which epsilon decays to min.")
    p.add_argument("--epsilon_min", type=float, default=0.05)
    p.add_argument("--buffer_size", type=int, default=50_000)
    p.add_argument("--target_update_freq", type=int, default=500)
    p.add_argument(
        "--grad_steps",
        type=int,
        default=8,
        help="SGD updates per env transition per agent (after replay has at least one batch).",
    )
    p.add_argument(
        "--eval_every",
        type=int,
        default=EXPERT_EVAL_EVERY_EPISODES,
        help="Run greedy eval every N training episodes (0 disables).",
    )
    p.add_argument("--eval_rollouts", type=int, default=30, help="Greedy episodes averaged for eval.")
    p.add_argument("--metrics_csv", type=str, default="artifacts/expert/expert_training_metrics.csv")
    p.add_argument(
        "--out",
        type=str,
        default="artifacts/expert/dqn_weights_agent_0.pt",
        help="Where to save agent_0 Q-network weights (FrozenTaskExpert checkpoint).",
    )
    return p


def run_expert_training(args) -> None:
    n_agents = args.n_agents
    num_envs = args.num_envs
    if n_agents <= 0:
        raise ValueError("--n_agents must be positive")
    if num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.grad_steps <= 0:
        raise ValueError("--grad_steps must be positive")

    print(f"Swarm RL device: {DEVICE}")

    vec_env = RationalSwarmForagingVecEnv(
        num_envs,
        n_agents=n_agents,
        grid_size=GRID_SIZE,
        num_food=NUM_FOOD,
        local_grid_size=5,
    )
    env0_agents = vec_env.possible_agents

    policies = {
        agent_id: DQNAgent(
            obs_dim=OBS_DIM,
            n_actions=N_ACTIONS_FULL,
            lr=args.alpha,
            gamma=args.gamma,
            epsilon_start=1.0,
            epsilon_end=args.epsilon_min,
            epsilon_decay_steps=args.epsilon_decay,
            buffer_size=args.buffer_size,
            batch_size=64,
            target_update_freq=args.target_update_freq,
            use_double_dqn=True,
            use_huber=True,
        )
        for agent_id in env0_agents
    }

    eval_env = RationalSwarmForagingEnv(
        n_agents=n_agents, grid_size=GRID_SIZE, num_food=NUM_FOOD, local_grid_size=5, render_mode=None
    )

    episode_metrics = []
    episode_team_returns: list[float] = []
    best_eval_so_far: float | None = None
    out_path = Path(args.out)

    for episode in range(args.episodes):
        seeds = (episode * num_envs + np.arange(num_envs, dtype=np.int64)).reshape(-1)
        obs_dict, _infos_reset = vec_env.reset(seeds)

        ep_team_return = 0.0
        ep_per_agent = {agent_id: 0.0 for agent_id in env0_agents}
        policy_losses: list[float] = []
        step_count = 0

        while step_count < args.max_steps_per_episode:
            actions_vec: dict[str, np.ndarray] = {}
            for agent_id in env0_agents:
                policy = policies[agent_id]
                obs_batch = obs_dict[agent_id].astype(np.float32, copy=False)
                action_batch = policy.select_actions_batch(obs_batch)
                actions_vec[agent_id] = action_batch.astype(np.int32, copy=False)

            next_obs_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict = vec_env.step(actions_vec)

            env_reward_row = infos_dict[env0_agents[0]]["env_reward"]
            step_team_sum = float(np.sum(env_reward_row))
            truncated = (step_count + 1) >= args.max_steps_per_episode

            for env_idx in range(num_envs):
                for agent_id in env0_agents:
                    r = float(rewards_dict[agent_id][env_idx])
                    ep_per_agent[agent_id] += r / float(num_envs)
                    policy = policies[agent_id]
                    terminations = terminations_dict[agent_id]
                    truncations = truncations_dict[agent_id]
                    done = bool(terminations[env_idx] or truncations[env_idx] or truncated)
                    policy.store(
                        obs_dict[agent_id][env_idx],
                        int(actions_vec[agent_id][env_idx]),
                        r,
                        next_obs_dict[agent_id][env_idx],
                        done,
                    )

            for _ in range(num_envs * args.grad_steps):
                for agent_id in env0_agents:
                    loss = policies[agent_id].train_step()
                    if loss is not None:
                        policy_losses.append(loss)

            ep_team_return += step_team_sum / float(num_envs)
            obs_dict = next_obs_dict
            step_count += 1

        for agent_id in env0_agents:
            policies[agent_id].advance_epsilon_episode()

        eps = policies[env0_agents[0]].epsilon()
        episode_team_returns.append(ep_team_return)
        eval_mean = float("nan")
        if args.eval_every > 0 and (episode + 1) % args.eval_every == 0:
            eval_mean = _eval_greedy_mean_team_return(
                eval_env, policies, args.max_steps_per_episode, args.eval_rollouts, episode
            )
            if best_eval_so_far is None or eval_mean > best_eval_so_far:
                best_eval_so_far = eval_mean
                _save_best_checkpoints(out_path, policies, list(env0_agents), n_agents)

        best_eval_csv = float(best_eval_so_far) if best_eval_so_far is not None else float("nan")

        row = {
            "episode": episode + 1,
            "n_agents": float(n_agents),
            "episode_mean_team_env_return": ep_team_return,
            "epsilon": float(eps),
            "policy_loss_mean": float(np.mean(policy_losses)) if policy_losses else float("nan"),
            "avg_mean_team_env_return_last_50": float(np.mean(episode_team_returns[-50:])),
            "eval_greedy_mean_team_return": eval_mean,
            "best_eval_greedy_mean_team_return": best_eval_csv,
        }
        for agent_id in env0_agents:
            row[f"return_{agent_id}"] = ep_per_agent[agent_id]
        episode_metrics.append(row)

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_team_returns[-50:])
            extra_eval = f" | eval greedy mean: {eval_mean:.3f}" if eval_mean == eval_mean else ""
            extra_best = f" | best eval: {best_eval_so_far:.3f}" if best_eval_so_far is not None else ""
            print(
                f"Episode {episode + 1}/{args.episodes} | Avg mean team env return (last 50 ep): {avg:.3f} | epsilon: {eps:.3f}"
                f"{extra_eval}{extra_best}"
            )

    vec_env.close()
    eval_env.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policies["agent_0"].q_net.state_dict(), out_path)
    if n_agents > 1:
        for agent_id in env0_agents:
            if agent_id == "agent_0":
                continue
            side = out_path.parent / f"dqn_weights_{agent_id}.pt"
            torch.save(policies[agent_id].q_net.state_dict(), side)
    print(f"Saved expert checkpoint to {out_path}")
    if best_eval_so_far is not None:
        best_path = _best_weight_path(out_path, "agent_0")
        print(f"Best eval greedy mean team return: {best_eval_so_far:.3f} (weights: {best_path})")

    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(episode_metrics)
    print(f"Saved metrics to {metrics_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_expert_training(args)


if __name__ == "__main__":
    main()
