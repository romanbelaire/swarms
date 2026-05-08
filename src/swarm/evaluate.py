import sys
from pathlib import Path

if __package__ is None:
    _src = Path(__file__).resolve().parents[1]
    _src_s = str(_src)
    if _src_s not in sys.path:
        sys.path.insert(0, _src_s)

import argparse
import torch

from swarm.config import GRID_SIZE, NUM_FOOD, OBS_DIM, N_ACTIONS_FULL
from swarm.env import RationalSwarmForagingEnv
from swarm.agents import DQNAgent, DEVICE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--max_steps_per_episode", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--weights_dir", type=str, default="artifacts", help="Directory with dqn_weights_agent_*.pt files")
    parser.add_argument(
        "--grid_size",
        type=int,
        default=GRID_SIZE,
        help="Must match expert training (default: swarm.config.GRID_SIZE).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    env = RationalSwarmForagingEnv(
        n_agents=args.n_agents,
        grid_size=args.grid_size,
        num_food=NUM_FOOD,
        local_grid_size=5,
        render_mode=None,
    )
    policies = {agent_id: DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS_FULL) for agent_id in env.possible_agents}
    for agent_id in env.possible_agents:
        path = f"{args.weights_dir}/dqn_weights_{agent_id}.pt"
        try:
            policies[agent_id].q_net.load_state_dict(torch.load(path, map_location=DEVICE))
            print(f"Loaded {path}")
        except FileNotFoundError:
            print(f"No {path} - using random policy")

    total_env_reward = 0.0
    total_steps = 0
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=episode)
        episode_env_reward = 0.0
        for _step in range(args.max_steps_per_episode):
            actions = {a: policies[a].select_action(obs[a], deterministic=True) for a in env.agents}
            obs, _rewards, term, trunc, infos = env.step(actions)
            episode_env_reward += infos[env.agents[0]]["env_reward"]
            total_steps += 1
            if all(term[a] or trunc[a] for a in env.agents):
                break
        total_env_reward += episode_env_reward

    env.close()
    print(f"Total env reward over {args.episodes} episodes ({total_steps} steps): {total_env_reward:.1f}")


if __name__ == "__main__":
    main()

