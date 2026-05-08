import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import torch

if __package__ is None:
    _src = Path(__file__).resolve().parents[1]
    _src_s = str(_src)
    if _src_s not in sys.path:
        sys.path.insert(0, _src_s)

from swarm.config import OBS_DIM, N_ACTIONS_FULL
from swarm.dr_panel import render_dr_panel
from swarm.env import RationalSwarmForagingEnv
from swarm.agents import DQNAgent, DRMixtureEvaluator, DEVICE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=7)
    parser.add_argument("--num_food", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=500, help="Steps per showcase run (or until quit)")
    parser.add_argument("--step_delay", type=float, default=0.05, help="Seconds between steps (0 for max speed)")
    parser.add_argument("--weights_dir", type=str, default="artifacts", help="Directory with dqn_weights_agent_*.pt files")
    parser.add_argument(
        "--show_dr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show DR mixture side panel when n_agents>1 (requires dr_mixture_*.pt and dr_mixture_meta.json). Ignored for n_agents=1 (base policy only).",
    )
    parser.add_argument(
        "--dr_weights_dir",
        type=str,
        default=None,
        help="Directory with dr_mixture_agent_*.pt and dr_mixture_meta.json (defaults to --weights_dir)",
    )
    parser.add_argument("--panel_width", type=int, default=320, help="Width of the DR side panel in pixels")
    return parser


def _rgb_array_to_surface(rgb: np.ndarray) -> pygame.Surface:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB array (H,W,3), got shape {rgb.shape}")
    rgb_u8 = np.ascontiguousarray(rgb.astype(np.uint8, copy=False))
    h, w, _ = rgb_u8.shape
    return pygame.image.frombuffer(rgb_u8.tobytes(), (w, h), "RGB")


def main():
    args = build_parser().parse_args()
    dr_dir = Path(args.dr_weights_dir) if args.dr_weights_dir is not None else Path(args.weights_dir)
    weights_dir = Path(args.weights_dir)

    use_dr_panel = args.show_dr and args.n_agents > 1

    if use_dr_panel:
        pygame.init()
        pygame.display.set_caption("Rational Swarm — DR mixture play")

        env = RationalSwarmForagingEnv(
            n_agents=args.n_agents,
            grid_size=args.grid_size,
            num_food=args.num_food,
            local_grid_size=5,
            render_mode="rgb_array",
        )
        meta_path = dr_dir / "dr_mixture_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"DR meta not found: {meta_path}")
        meta = DRMixtureEvaluator.load_meta(meta_path)
        full_duration = float(meta["full_duration"])
        agents_meta = meta["agents"]
        my_mean_by_agent = {}
        dr_evaluators = {}
        for agent_id in env.possible_agents:
            if agent_id not in agents_meta:
                raise KeyError(f"Agent {agent_id} missing from dr_mixture_meta.json agents")
            am = agents_meta[agent_id]
            cnt = int(am["running_own_p_count"])
            if cnt <= 0:
                raise ValueError(f"Invalid running_own_p_count for {agent_id}: {cnt}")
            my_mean_by_agent[agent_id] = float(am["running_own_p_sum"]) / float(cnt)
            dr_path = dr_dir / f"dr_mixture_{agent_id}.pt"
            if not dr_path.is_file():
                raise FileNotFoundError(f"DR mixture weights not found: {dr_path}")
            dr_evaluators[agent_id] = DRMixtureEvaluator(
                state_dict_path=dr_path,
                full_duration=full_duration,
                my_mean=my_mean_by_agent[agent_id],
                obs_dim=OBS_DIM,
            )

        policies = {agent_id: DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS_FULL) for agent_id in env.possible_agents}
        for agent_id in env.possible_agents:
            path = weights_dir / f"dqn_weights_{agent_id}.pt"
            policies[agent_id].q_net.load_state_dict(torch.load(path, map_location=DEVICE))
            policies[agent_id].epsilon_episodes = policies[agent_id].epsilon_decay_steps
            print(f"Loaded {path}")

        grid_px = env.window_size
        panel_w = int(args.panel_width)
        min_panel_body = 40 + len(env.possible_agents) * 185
        win_h = max(grid_px, min_panel_body)
        win_w = grid_px + panel_w
        screen = pygame.display.set_mode((win_w, win_h))
        clock = pygame.time.Clock()

        obs, _ = env.reset(seed=0)
        running = True
        step_count = 0
        total_env_reward = 0.0
        last_team_util = 0.0
        last_env_r = 0.0

        def build_rows(o_dict, team_util: float, infos_dict: dict):
            rows = []
            for aid in env.possible_agents:
                own_p = float(infos_dict[aid]["p_t"])
                out = dr_evaluators[aid].evaluate(o_dict[aid], team_util, own_p)
                rows.append(
                    {
                        "agent_id": aid,
                        "u": out["u"],
                        "v": out["v"],
                        "w": out["w"],
                        "scenario_delta": out["scenario_delta"],
                        "mixed_reward": out["mixed_reward"],
                    }
                )
            return rows

        print("Showcase running (DR panel). Close window or press ESC to quit.")
        while running and step_count < args.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            if not running:
                break

            actions = {a: policies[a].select_action(obs[a], deterministic=True) for a in env.possible_agents}
            obs, _rewards, _term, _trunc, infos = env.step(actions)
            last_env_r = float(infos[env.possible_agents[0]]["env_reward"])
            total_env_reward += last_env_r
            last_team_util = float(
                sum(float(infos[aid]["p_t"]) - float(infos[aid]["c_t"]) for aid in env.possible_agents)
            )
            per_rows = build_rows(obs, last_team_util, infos)
            step_count += 1

            rgb = env.render()
            if rgb is None:
                raise ValueError("env.render() returned None in rgb_array mode")
            env_surf = _rgb_array_to_surface(rgb)
            screen.fill((40, 40, 45))
            screen.blit(env_surf, (0, 0))
            panel_rect = pygame.Rect(grid_px, 0, panel_w, win_h)
            render_dr_panel(screen, panel_rect, per_rows, last_env_r, last_team_util, step_count, my_mean_by_agent)
            pygame.display.flip()

            if args.step_delay > 0:
                time.sleep(args.step_delay)
            clock.tick(60)

        env.close()
        pygame.quit()
        print(f"Showcase ended after {step_count} steps. Total env reward: {total_env_reward:.1f}")
        return

    # Human-only mode (no DR panel)
    env = RationalSwarmForagingEnv(
        n_agents=args.n_agents,
        grid_size=args.grid_size,
        num_food=args.num_food,
        local_grid_size=5,
        render_mode="human",
    )
    policies = {agent_id: DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS_FULL) for agent_id in env.possible_agents}
    for agent_id in env.possible_agents:
        best_path = weights_dir / f"dqn_weights_{agent_id}_best.pt"
        path = best_path if best_path.is_file() else weights_dir / f"dqn_weights_{agent_id}.pt"
        policies[agent_id].q_net.load_state_dict(torch.load(path, map_location=DEVICE))
        # Fresh DQNAgent has epsilon_episodes=0 -> epsilon() is 1.0; without this, deterministic=False is uniform random.
        policies[agent_id].epsilon_episodes = policies[agent_id].epsilon_decay_steps
        print(f"Loaded {path}")

    obs, _ = env.reset(seed=0)
    env.render()
    running = True
    step_count = 0
    total_env_reward = 0.0
    print("Showcase running (policy steps automatically). Close window or press ESC to quit.")
    while running and step_count < args.max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        if not running:
            break
        actions = {a: policies[a].select_action(obs[a], deterministic=False) for a in env.possible_agents}
        obs, _rewards, _term, _trunc, infos = env.step(actions)
        env.render()
        total_env_reward += infos[env.possible_agents[0]]["env_reward"]
        step_count += 1
        if args.step_delay > 0:
            time.sleep(args.step_delay)
    env.close()
    print(f"Showcase ended after {step_count} steps. Total env reward: {total_env_reward:.1f}")


if __name__ == "__main__":
    main()
