import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch

if __package__ is None:
    _src = Path(__file__).resolve().parents[2]
    _src_s = str(_src)
    if _src_s not in sys.path:
        sys.path.insert(0, _src_s)

from swarm.env import RationalSwarmForagingEnv
from swarm.conflict_instances import (
    ClosedConflictInstance,
    ConflictInstanceTracker,
    episode_mean_own_p_norm,
    episode_mean_own_p_norm_all_envs,
    episode_mean_team_util_norm,
    episode_mean_team_util_norm_all_envs,
    new_conflict_trackers,
    step_team_utility_mean,
)
from swarm.agents import DQNAgent, DEVICE, FrozenTaskExpert, PCCriticLearner, DRScenarioMixtureLearner, UCB1Bandit, DRUCBPolicyLearner
from swarm.config import (
    N_EPISODES,
    MAX_STEPS_PER_EPISODE,
    GRID_SIZE,
    NUM_FOOD,
    LOCAL_GRID_SIZE,
    OBS_DIM,
    N_ACTIONS_FULL,
    N_ACTIONS_TASK_AVOID,
    TASK_POLICY_ACTION,
    NO_OP_3_TURNS_ACTION,
    MOVE_BACKWARDS_2_ACTION,
    RANDOM_WALK_3_ACTION,
    WAIT2_THEN_FORWARD1_ACTION,
    MOVE_CLEAR_ACTION,
    HANDSHAKE_ACTION,
    RESERVE_PARITY_ACTION,
    RESERVE_PARITY_ESCAPE_ACTION,
    TASK_AVOID_ENABLED_ACTION_IDS,
    MAX_CONFLICT_STEPS,
    MAX_CONFLICT_STEPS_TYPE,
    FULL_DURATION_NORM,
)

CONFLICT_ACTION_NAMES = [
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
    "freeze_tag",
]
ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
PRIORITY_SWAP_TOKEN_STEPS = 3
CONFLICT_ACTION_ALIASES = {
    "swap_food_n3": "pass_food_n3",
    "wait_3": "wait3",
    "backward_3": "backward3",
    "backwards_2": "backward3",
    "backwards_3": "backward3",
}
BANDIT_REWARD_MODEL_NAMES = [
    "solver_allc",
    "solver_allp",
    "solver_allsame",
    "neutral_allc",
    "neutral_allp",
    "neutral_allsame",
    "causer_allc",
    "causer_allp",
    "causer_allsame",
]
BANDIT_CREDIT_MODE_NAMES = [
    "instance_credited",
    "episode_shared",
    "per_arm_credited",
    "arm_relative_my_role",
    "step_level",
]


def invert_move(action: int) -> int:
    if action == 0:
        return 1
    if action == 1:
        return 0
    if action == 2:
        return 3
    if action == 3:
        return 2
    return 4


def random_walk_actions(length: int) -> list[int]:
    return [int(np.random.randint(0, 4)) for _ in range(length)]


def move_clear_action(obs: np.ndarray) -> int:
    n_tiles = obs.shape[0] - 3
    local_grid_size = int(np.sqrt(n_tiles))
    grid = obs[:n_tiles].reshape(local_grid_size, local_grid_size)
    c = local_grid_size // 2
    directions = [
        (0, -1, 0),  # up
        (1, 1, 0),   # down
        (2, 0, -1),  # left
        (3, 0, 1),   # right
    ]
    best_action = 0
    best_score = 10_000
    for action, dx, dy in directions:
        obstacles = 0
        for step in range(1, c + 1):
            cell = grid[c + dx * step, c + dy * step]
            if cell == 3 or cell == 4 or cell == 5:
                obstacles += 1
        if obstacles < best_score:
            best_score = obstacles
            best_action = action
    return best_action


def forward_target_agent(agent_id: str, positions: dict[str, list[int]], heading_action: int) -> str | None:
    dx, dy = ACTION_DELTAS[heading_action]
    x, y = positions[agent_id]
    tx, ty = x + dx, y + dy
    for other_id, other_pos in positions.items():
        if other_id == agent_id:
            continue
        if other_pos[0] == tx and other_pos[1] == ty:
            return other_id
    return None


def adjacent_agents(agent_id: str, positions: dict[str, list[int]]) -> list[tuple[str, int]]:
    ax, ay = positions[agent_id]
    neighbors = []
    for other_id, other_pos in positions.items():
        if other_id == agent_id:
            continue
        ox, oy = other_pos
        if ox == ax - 1 and oy == ay:
            neighbors.append((other_id, 0))
        elif ox == ax + 1 and oy == ay:
            neighbors.append((other_id, 1))
        elif ox == ax and oy == ay - 1:
            neighbors.append((other_id, 2))
        elif ox == ax and oy == ay + 1:
            neighbors.append((other_id, 3))
    return neighbors


def resolve_handshake_step(
    handshake_selected: set[str],
    low_actions: dict[str, int],
    macro_queues: dict[str, list[int]],
    headings: dict[str, int],
    positions: dict[str, list[int]],
):
    mutual_pairs = []
    matched = set()
    for agent_id in handshake_selected:
        target = forward_target_agent(agent_id, positions, headings[agent_id])
        if target is None or target not in handshake_selected:
            continue
        target_target = forward_target_agent(target, positions, headings[target])
        if target_target != agent_id:
            continue
        key = tuple(sorted((agent_id, target)))
        if key in matched:
            continue
        matched.add(key)
        mutual_pairs.append((agent_id, target))

    succeeded = set()
    for agent_a, agent_b in mutual_pairs:
        low_actions[agent_a] = headings[agent_a]
        low_actions[agent_b] = headings[agent_b]
        macro_queues[agent_a] = [headings[agent_a]]
        macro_queues[agent_b] = [headings[agent_b]]
        succeeded.add(agent_a)
        succeeded.add(agent_b)

    for agent_id in handshake_selected:
        if agent_id in succeeded:
            continue
        neighbors = adjacent_agents(agent_id, positions)
        if len(neighbors) > 0:
            neighbor_idx = int(np.random.randint(0, len(neighbors)))
            headings[agent_id] = neighbors[neighbor_idx][1]
        low_actions[agent_id] = 4
        macro_queues[agent_id] = [4, 4]


def _freeze_tag_selected_components(freeze_selected: set[str], positions: dict[str, list[int]]) -> list[set[str]]:
    adjacency: dict[str, list[str]] = {agent_id: [] for agent_id in freeze_selected}
    for aid in freeze_selected:
        for other_id, _ in adjacent_agents(aid, positions):
            if other_id in freeze_selected:
                adjacency[aid].append(other_id)
    visited: set[str] = set()
    components: list[set[str]] = []
    for start in sorted(freeze_selected):
        if start in visited:
            continue
        stack = [start]
        comp: set[str] = set()
        while len(stack) > 0:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.add(u)
            for v in adjacency[u]:
                if v not in visited:
                    stack.append(v)
        components.append(comp)
    return components


def resolve_freeze_tag_step(
    freeze_selected: set[str],
    env: RationalSwarmForagingEnv,
    expert_actions_subset: dict[str, int],
    low_actions: dict[str, int],
    macro_queues: dict[str, list[int]],
    positions: dict[str, list[int]],
):
    succeeded = set()
    for comp in _freeze_tag_selected_components(freeze_selected, positions):
        if len(comp) < 3:
            continue
        env.grant_immunity_unlock_bulk(tuple(sorted(comp)))
        for aid in comp:
            low_actions[aid] = int(expert_actions_subset[aid])
            macro_queues[aid] = []
        succeeded.update(comp)

    sorted_sel = sorted(freeze_selected)
    for aid in sorted_sel:
        if aid in succeeded:
            continue
        partner_candidates = sorted(
            [nid for nid, _ in adjacent_agents(aid, positions) if nid in freeze_selected and nid not in succeeded]
        )
        if len(partner_candidates) == 0:
            continue
        partner = partner_candidates[0]
        env.apply_freeze_tag_pair(aid, partner)
        low_actions[aid] = int(expert_actions_subset[aid])
        low_actions[partner] = int(expert_actions_subset[partner])
        macro_queues[aid] = []
        macro_queues[partner] = []
        succeeded.add(aid)
        succeeded.add(partner)
    for agent_id in freeze_selected:
        if agent_id in succeeded:
            continue
        low_actions[agent_id] = 4
        macro_queues[agent_id] = [4, 4]


def _bounded_target_from_action(position: list[int], action: int, grid_size: int) -> tuple[int, int]:
    x, y = position
    if action == 0 and x > 0:
        return x - 1, y
    if action == 1 and x < grid_size - 1:
        return x + 1, y
    if action == 2 and y > 0:
        return x, y - 1
    if action == 3 and y < grid_size - 1:
        return x, y + 1
    return x, y


def _parity_priority(position: list[int]) -> int:
    x, y = position
    return int((x % 2) == (y % 2))


def _preferred_escape_actions(heading_action: int) -> list[int]:
    if heading_action == 0 or heading_action == 1:
        return [2, 3, invert_move(heading_action), heading_action]
    return [0, 1, invert_move(heading_action), heading_action]


def _pick_escape_action(
    *,
    agent_id: str,
    heading_action: int,
    positions: dict[str, list[int]],
    grid_size: int,
    blocked_targets: set[tuple[int, int]],
    occupancy: dict[tuple[int, int], int],
) -> int:
    candidates = _preferred_escape_actions(heading_action)
    best_action = candidates[0]
    best_score = 10_000
    for action in candidates:
        target = _bounded_target_from_action(positions[agent_id], action, grid_size)
        blocked_penalty = 1 if target in blocked_targets else 0
        score = blocked_penalty * 1000 + occupancy[target]
        if score < best_score:
            best_score = score
            best_action = action
    return best_action


def _manhattan_distance(position: list[int], base_position: tuple[int, int]) -> int:
    return abs(position[0] - base_position[0]) + abs(position[1] - base_position[1])


def resolve_reserve_parity_step(
    reserve_selected: set[str],
    low_actions: dict[str, int],
    macro_queues: dict[str, list[int]],
    headings: dict[str, int],
    positions: dict[str, list[int]],
    grid_size: int,
):
    proposed_actions: dict[str, int] = {}
    target_groups: dict[tuple[int, int], list[str]] = {}
    for agent_id in reserve_selected:
        action = headings[agent_id]
        proposed_actions[agent_id] = action
        target = _bounded_target_from_action(positions[agent_id], action, grid_size)
        if target not in target_groups:
            target_groups[target] = []
        target_groups[target].append(agent_id)

    winners = set()
    for claimers in target_groups.values():
        winner = sorted(claimers, key=lambda aid: (-_parity_priority(positions[aid]), aid))[0]
        winners.add(winner)

    for agent_id in reserve_selected:
        low_actions[agent_id] = 4
        move_action = proposed_actions[agent_id]
        if agent_id in winners:
            macro_queues[agent_id] = [move_action]
        else:
            macro_queues[agent_id] = [4, move_action]


def resolve_reserve_parity_escape_step(
    reserve_selected: set[str],
    low_actions: dict[str, int],
    macro_queues: dict[str, list[int]],
    headings: dict[str, int],
    positions: dict[str, list[int]],
    grid_size: int,
):
    proposed_actions: dict[str, int] = {}
    target_groups: dict[tuple[int, int], list[str]] = {}
    occupancy: dict[tuple[int, int], int] = {
        (x, y): 0
        for x in range(grid_size)
        for y in range(grid_size)
    }
    for pos in positions.values():
        key = (pos[0], pos[1])
        occupancy[key] += 1
    for agent_id in reserve_selected:
        action = headings[agent_id]
        proposed_actions[agent_id] = action
        target = _bounded_target_from_action(positions[agent_id], action, grid_size)
        if target not in target_groups:
            target_groups[target] = []
        target_groups[target].append(agent_id)
    winners = set()
    for claimers in target_groups.values():
        winner = sorted(claimers, key=lambda aid: (-_parity_priority(positions[aid]), aid))[0]
        winners.add(winner)
    winner_targets = {
        _bounded_target_from_action(positions[agent_id], proposed_actions[agent_id], grid_size)
        for agent_id in winners
    }
    for agent_id in reserve_selected:
        low_actions[agent_id] = 4
        move_action = proposed_actions[agent_id]
        if agent_id in winners:
            macro_queues[agent_id] = [move_action]
            continue
        escape_action = _pick_escape_action(
            agent_id=agent_id,
            heading_action=move_action,
            positions=positions,
            grid_size=grid_size,
            blocked_targets=winner_targets,
            occupancy=occupancy,
        )
        macro_queues[agent_id] = [escape_action]


def resolve_pass_food_n3_step(
    pass_selected: set[str],
    low_actions: dict[str, int],
    macro_queues: dict[str, list[int]],
    positions: dict[str, list[int]],
    agent_holding_food: dict[str, bool],
    base_position: tuple[int, int],
    obs_by_agent: dict[str, np.ndarray],
):
    proposals_by_receiver: dict[str, list[str]] = {}
    for sender_id in pass_selected:
        if not agent_holding_food[sender_id]:
            continue
        sender_dist = _manhattan_distance(positions[sender_id], base_position)
        receiver_candidates = []
        for neighbor_id, _ in adjacent_agents(sender_id, positions):
            if neighbor_id not in pass_selected:
                continue
            if agent_holding_food[neighbor_id]:
                continue
            receiver_dist = _manhattan_distance(positions[neighbor_id], base_position)
            if receiver_dist < sender_dist:
                receiver_candidates.append((receiver_dist, neighbor_id))
        if len(receiver_candidates) == 0:
            continue
        receiver_candidates.sort(key=lambda item: (item[0], item[1]))
        best_dist = receiver_candidates[0][0]
        best_receivers = [receiver_id for dist, receiver_id in receiver_candidates if dist == best_dist]
        chosen_receiver = best_receivers[int(np.random.randint(0, len(best_receivers)))]
        if chosen_receiver not in proposals_by_receiver:
            proposals_by_receiver[chosen_receiver] = []
        proposals_by_receiver[chosen_receiver].append(sender_id)

    for receiver_id, sender_ids in proposals_by_receiver.items():
        chosen_sender = sender_ids[int(np.random.randint(0, len(sender_ids)))]
        agent_holding_food[chosen_sender] = False
        agent_holding_food[receiver_id] = True

    for agent_id in pass_selected:
        n_adjacent_agents = len(adjacent_agents(agent_id, positions))
        mc = move_clear_action(obs_by_agent[agent_id])
        low_actions[agent_id] = 4
        if n_adjacent_agents >= 2:
            macro_queues[agent_id] = [mc]
        else:
            macro_queues[agent_id] = [4, 4, mc]


def resolve_priority_swap_selection_step(
    priority_selected: set[str],
    low_actions: dict[str, int],
    macro_queues: dict[str, list[int]],
    expert_actions_by_agent: dict[str, int],
    positions: dict[str, list[int]],
    token_remaining: dict[str, int],
    grid_size: int,
):
    if len(priority_selected) == 0:
        return
    target_groups: dict[tuple[int, int], list[str]] = {}
    for agent_id in priority_selected:
        expert_action = expert_actions_by_agent[agent_id]
        target = _bounded_target_from_action(positions[agent_id], expert_action, grid_size)
        if target not in target_groups:
            target_groups[target] = []
        target_groups[target].append(agent_id)
    winners = set()
    for claimers in target_groups.values():
        winner = sorted(claimers, key=lambda aid: (-_parity_priority(positions[aid]), aid))[0]
        winners.add(winner)
    for agent_id in priority_selected:
        low_actions[agent_id] = 4
        macro_queues[agent_id] = [4, 4]
        if agent_id in winners:
            low_actions[agent_id] = int(expert_actions_by_agent[agent_id])
            macro_queues[agent_id] = []
            token_remaining[agent_id] = PRIORITY_SWAP_TOKEN_STEPS


def resolve_priority_swap_token_step(
    env_actions: dict[str, int],
    macro_queues: dict[str, list[int]],
    prev_positions: dict[str, list[int]],
    prev_conflict_flags: dict[str, bool],
    token_remaining: dict[str, int],
    grid_size: int,
):
    token_holders = [agent_id for agent_id, remaining in token_remaining.items() if remaining > 0]
    if len(token_holders) == 0:
        return
    holder_targets: dict[tuple[int, int], list[str]] = {}
    for agent_id in token_holders:
        action = env_actions[agent_id]
        if action < 0 or action > 3:
            continue
        target = _bounded_target_from_action(prev_positions[agent_id], action, grid_size)
        if target not in holder_targets:
            holder_targets[target] = []
        holder_targets[target].append(agent_id)
    losing_holders = set()
    for claimers in holder_targets.values():
        if len(claimers) <= 1:
            continue
        winner = sorted(claimers, key=lambda aid: (-token_remaining[aid], aid))[0]
        for agent_id in claimers:
            if agent_id != winner:
                losing_holders.add(agent_id)
    for agent_id in losing_holders:
        env_actions[agent_id] = 4
        macro_queues[agent_id] = [4, 4]
    target_to_agent = {(pos[0], pos[1]): agent_id for agent_id, pos in prev_positions.items()}
    blocker_claimed = set()
    for agent_id in token_holders:
        if agent_id in losing_holders:
            continue
        action = env_actions[agent_id]
        if action < 0 or action > 3:
            continue
        target = _bounded_target_from_action(prev_positions[agent_id], action, grid_size)
        if target not in target_to_agent:
            continue
        blocker = target_to_agent[target]
        if blocker == agent_id:
            continue
        if not prev_conflict_flags[blocker]:
            continue
        if blocker in blocker_claimed:
            continue
        blocker_claimed.add(blocker)
        env_actions[blocker] = invert_move(action)


def update_headings_from_step(
    headings: dict[str, int],
    prev_positions: dict[str, list[int]],
    new_positions: dict[str, list[int]],
    applied_actions: dict[str, int],
):
    for agent_id, action in applied_actions.items():
        if action < 0 or action > 3:
            continue
        prev_pos = prev_positions[agent_id]
        new_pos = new_positions[agent_id]
        if prev_pos[0] != new_pos[0] or prev_pos[1] != new_pos[1]:
            headings[agent_id] = action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of synchronous environment replicas")
    parser.add_argument("--episodes", type=int, default=N_EPISODES, help="Number of episodes")
    parser.add_argument("--max_steps_per_episode", type=int, default=MAX_STEPS_PER_EPISODE, help="Max steps per episode")
    parser.add_argument("--grid_size", type=int, default=GRID_SIZE, help="Environment grid size")
    parser.add_argument("--num_food", type=int, default=NUM_FOOD, help="Number of food pellets present")
    parser.add_argument("--local_grid_size", type=int, default=LOCAL_GRID_SIZE, help="Local observation window size")
    parser.add_argument("--alpha", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor")
    parser.add_argument("--epsilon_decay", type=int, default=1000, help="Steps over which epsilon decays to min")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon (exploration rate)")
    parser.add_argument("--buffer_size", type=int, default=25000, help="Replay buffer size")
    parser.add_argument("--target_update_freq", type=int, default=3000, help="Target network update frequency")
    parser.add_argument("--load_weights", type=str, default=None, help="Path to single-agent weights to initialize all agents")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "task_avoid"], help="Training mode")
    parser.add_argument("--expert_checkpoint", type=str, default=None, help="Single-agent checkpoint used as frozen expert")
    parser.add_argument("--critic_lr", type=float, default=5e-4, help="Learning rate for P,C critics")
    parser.add_argument("--critic_gamma", type=float, default=0.99, help="TD gamma for critic rate targets")
    parser.add_argument("--critic_buffer_size", type=int, default=25000, help="Replay buffer size for critics")
    parser.add_argument("--critic_target_update_freq", type=int, default=3000, help="Hard target network update frequency for critics")
    parser.add_argument("--train_every", type=int, default=1, help="Deprecated flag preserved for compatibility")
    parser.add_argument("--reward_mode", type=str, default="mean_DR", choices=["mean_DR", "scenario_mixture"], help="Reward shaping mode")
    parser.add_argument("--dr_lr", type=float, default=5e-4, help="Learning rate for scenario-mixture reward learner")
    parser.add_argument("--dr_entropy_coef", type=float, default=0.0, help="Entropy regularization for scenario-mixture gates")
    parser.add_argument("--dr_my_mean_mode", type=str, default="running_own_p_mean", choices=["running_own_p_mean"])
    parser.add_argument("--metrics_csv", type=str, default="artifacts/training_metrics.csv", help="Output CSV path for per-episode metrics.")
    parser.add_argument(
        "--baseline_mode",
        type=str,
        default="none",
        choices=["none", "bandit_ucb1", "dr_ucb_mixture", "random_conflict", "fixed_conflict", "collision_free"],
        help="Ablation baseline mode. 'none' runs learned DQN pipeline. "
        "'collision_free' runs expert-task actions only with inter-agent positional collisions disabled.",
    )
    parser.add_argument(
        "--fixed_conflict_action",
        type=str,
        default="backward3",
        choices=CONFLICT_ACTION_NAMES,
        help="Fixed conflict macro action used when --baseline_mode fixed_conflict.",
    )
    parser.add_argument(
        "--bandit_reward_model",
        type=str,
        default="neutral_allsame",
        choices=BANDIT_REWARD_MODEL_NAMES,
        help="DR reward model used for UCB1 bandit updates in baseline mode.",
    )
    parser.add_argument(
        "--bandit_conflict_arms",
        type=str,
        default="",
        help=(
            "bandit_ucb1 only: comma-separated subset of conflict macros "
            "(default empty = full list CONFLICT_ACTION_NAMES). "
            'Example: "randomwalk3,freeze_tag,wait3,move_clear,backward3". Aliases: wait_3, backwards_2.'
        ),
    )
    parser.add_argument(
        "--bandit_credit_mode",
        type=str,
        default="instance_credited",
        choices=BANDIT_CREDIT_MODE_NAMES,
        help="How bandit_ucb1 assigns credit: instance_credited = one softplus(D) update per closed conflict window for the active arm.",
    )
    parser.add_argument("--dr_meta_lr", type=float, default=1e-3, help="Learning rate for dr_ucb_mixture meta-policy MLP.")
    parser.add_argument("--dr_meta_batch_size", type=int, default=32, help="REINFORCE batch size for dr_ucb_mixture meta-policy.")
    parser.add_argument(
        "--max_conflict_steps",
        type=int,
        default=MAX_CONFLICT_STEPS,
        help="Max conflict-instance window (0 = no cap). time: p+c steps; distance: unmoved steps.",
    )
    parser.add_argument(
        "--max_conflict_steps_type",
        type=str,
        default=MAX_CONFLICT_STEPS_TYPE,
        choices=["time", "distance"],
        help="How max_conflict_steps is measured for forced instance splits.",
    )
    return parser


def _agent_moved(prev_position: list[int], new_position: list[int]) -> bool:
    return prev_position[0] != new_position[0] or prev_position[1] != new_position[1]


def _macro_action_sequence(conflict_action_name: str, expert_action: int, obs: np.ndarray) -> list[int]:
    if conflict_action_name == "wait3":
        return [4, 4, 4]
    if conflict_action_name == "backward3":
        b = invert_move(int(expert_action))
        return [b, b, b]
    if conflict_action_name == "randomwalk3":
        return random_walk_actions(3)
    if conflict_action_name == "wait2_forward1":
        return [4, 4, int(expert_action)]
    if conflict_action_name == "move_clear":
        return [move_clear_action(obs)]
    if conflict_action_name == "handshake":
        raise ValueError("handshake must be resolved jointly at step level")
    if conflict_action_name == "reserve_parity":
        raise ValueError("reserve_parity must be resolved jointly at step level")
    if conflict_action_name == "reserve_parity_escape":
        raise ValueError("reserve_parity_escape must be resolved jointly at step level")
    if conflict_action_name == "priority_swap_n3":
        raise ValueError("priority_swap_n3 must be resolved jointly at step level")
    if conflict_action_name == "pass_food_n3":
        raise ValueError("pass_food_n3 must be resolved jointly at step level")
    if conflict_action_name == "freeze_tag":
        raise ValueError("freeze_tag must be resolved jointly at step level")
    raise ValueError(f"Unknown conflict action name: {conflict_action_name}")


def _arm_to_name(arm: int) -> str:
    arm_names = CONFLICT_ACTION_NAMES
    if arm < 0 or arm >= len(arm_names):
        raise ValueError(f"Invalid arm: {arm}")
    return arm_names[arm]


def _canonical_conflict_action_name(action_name: str) -> str:
    if action_name in CONFLICT_ACTION_ALIASES:
        return CONFLICT_ACTION_ALIASES[action_name]
    return action_name


def baseline_bandit_arms_tuple(bandit_conflict_arms_csv: str) -> tuple[str, ...]:
    if bandit_conflict_arms_csv.strip() == "":
        return tuple(CONFLICT_ACTION_NAMES)
    parsed: list[str] = []
    for raw in bandit_conflict_arms_csv.split(","):
        token = raw.strip()
        if token == "":
            continue
        name = _canonical_conflict_action_name(token)
        if name not in CONFLICT_ACTION_NAMES:
            raise ValueError(f"Unknown bandit conflict arm '{token}' (canonical '{name}'). Valid: {CONFLICT_ACTION_NAMES}")
        parsed.append(name)
    uniq = tuple(dict.fromkeys(parsed))
    if len(uniq) < 2:
        raise ValueError(f"bandit_conflict_arms must name at least 2 distinct macros; got {list(uniq)}")
    return uniq


def _dr_softplus(x: float) -> float:
    x = float(x)
    if x > 20.0:
        return x
    if x < -20.0:
        return float(np.exp(x))
    return float(np.log1p(np.exp(x)))


def _bandit_dr_delta_raw(
    model_name: str,
    *,
    team_utility_mean: float,
    own_p_mean: float,
) -> float:
    fd = float(FULL_DURATION_NORM)
    if model_name == "solver_allc":
        return 0.0
    if model_name == "solver_allp":
        return fd
    if model_name == "solver_allsame":
        return float(team_utility_mean)
    if model_name == "neutral_allc":
        return float(-own_p_mean)
    if model_name == "neutral_allp":
        return float(fd - own_p_mean)
    if model_name == "neutral_allsame":
        return float(team_utility_mean - own_p_mean)
    if model_name == "causer_allc":
        return float(-fd)
    if model_name == "causer_allp":
        return 0.0
    if model_name == "causer_allsame":
        return float(team_utility_mean - fd)
    raise ValueError(f"Unknown bandit_reward_model: {model_name}")


def _bandit_reward_from_episode(
    model_name: str,
    *,
    team_utility_mean: float,
    own_p_mean: float,
) -> float:
    return _dr_softplus(
        _bandit_dr_delta_raw(
            model_name,
            team_utility_mean=team_utility_mean,
            own_p_mean=own_p_mean,
        )
    )


def _new_episode_arm_stats(agent_ids: list[str], arm_names: tuple[str, ...]) -> dict[str, dict[str, dict[str, list[float]]]]:
    return {
        agent_id: {arm_name: {"own_p": [], "team_util": []} for arm_name in arm_names}
        for agent_id in agent_ids
    }


def _mean_arm_stat(samples: list[float]) -> float:
    return float(np.mean(samples))


def _new_training_arm_samples(bandit_arm_names: tuple[str, ...]) -> dict[str, dict[str, list[float]]]:
    return {
        arm_name: {"reward": [], "delta": [], "own_p": [], "team_util": []}
        for arm_name in bandit_arm_names
    }


def _bandit_arm_metric_columns(
    *,
    bandit_arm_names: tuple[str, ...],
    bandits: dict[str, UCB1Bandit],
    agent_ids: list[str],
    training_arm_samples: dict[str, dict[str, list[float]]],
    episode_arm_counts: dict[str, int],
    arm_counts: dict[str, int],
) -> dict[str, float]:
    columns: dict[str, float] = {}
    ep_sel_total = sum(episode_arm_counts[name] for name in bandit_arm_names)
    cum_sel_total = sum(arm_counts[name] for name in bandit_arm_names)
    pull_counts = np.mean(np.stack([bandits[agent_id].counts for agent_id in agent_ids], axis=0), axis=0)
    for name in CONFLICT_ACTION_NAMES:
        for suffix in (
            "pull_count",
            "ep_sel_frac",
            "cum_sel_frac",
            "reward_mean",
            "reward_var",
            "delta_mean",
            "delta_var",
            "own_p_mean",
            "team_util_mean",
            "team_util_var",
        ):
            columns[f"arm_{name}_{suffix}"] = np.nan
    for arm_idx, name in enumerate(bandit_arm_names):
        columns[f"arm_{name}_pull_count"] = float(pull_counts[arm_idx])
        columns[f"arm_{name}_ep_sel_frac"] = (
            float(episode_arm_counts[name] / ep_sel_total) if ep_sel_total > 0 else np.nan
        )
        columns[f"arm_{name}_cum_sel_frac"] = (
            float(arm_counts[name] / cum_sel_total) if cum_sel_total > 0 else np.nan
        )
        samples = training_arm_samples[name]
        rewards = samples["reward"]
        deltas = samples["delta"]
        own_ps = samples["own_p"]
        team_utils = samples["team_util"]
        columns[f"arm_{name}_reward_mean"] = float(np.mean(rewards)) if len(rewards) > 0 else np.nan
        columns[f"arm_{name}_reward_var"] = float(np.var(rewards)) if len(rewards) > 0 else np.nan
        columns[f"arm_{name}_delta_mean"] = float(np.mean(deltas)) if len(deltas) > 0 else np.nan
        columns[f"arm_{name}_delta_var"] = float(np.var(deltas)) if len(deltas) > 0 else np.nan
        columns[f"arm_{name}_own_p_mean"] = float(np.mean(own_ps)) if len(own_ps) > 0 else np.nan
        columns[f"arm_{name}_team_util_mean"] = float(np.mean(team_utils)) if len(team_utils) > 0 else np.nan
        columns[f"arm_{name}_team_util_var"] = float(np.var(team_utils)) if len(team_utils) > 0 else np.nan
    return columns


def _record_closed_conflict_instance(
    *,
    closed: ClosedConflictInstance,
    episode_own_p_samples: list[float],
    episode_own_c_samples: list[float],
    episode_team_utility_samples: list[float],
    episode_own_p_by_agent: dict[str, list[float]],
    episode_arm_stats: dict[str, dict[str, dict[str, list[float]]]] | None,
    agent_id: str,
    arm_name: str | None,
):
    episode_own_p_samples.append(closed.p_norm)
    episode_own_c_samples.append(closed.c_norm)
    episode_team_utility_samples.append(closed.team_util_norm)
    episode_own_p_by_agent[agent_id].append(closed.p_norm)
    if episode_arm_stats is not None and arm_name is not None:
        episode_arm_stats[agent_id][arm_name]["own_p"].append(closed.p_norm)
        episode_arm_stats[agent_id][arm_name]["team_util"].append(closed.team_util_norm)


def _credit_bandit_conflict_instance(
    *,
    bandits: dict[str, UCB1Bandit],
    bandit_arm_names: tuple[str, ...],
    bandit_reward_model: str,
    agent_id: str,
    arm_name: str,
    closed: ClosedConflictInstance,
):
    arm_idx = bandit_arm_names.index(arm_name)
    reward = _bandit_reward_from_episode(
        bandit_reward_model,
        team_utility_mean=closed.team_util_norm,
        own_p_mean=closed.p_norm,
    )
    bandits[agent_id].update(arm_idx, reward)


def _credit_dr_ucb_mixture_instance(
    *,
    nested_bandits: dict[str, dict[str, UCB1Bandit]],
    dr_policies: dict[str, DRUCBPolicyLearner],
    bandit_arm_names: tuple[str, ...],
    agent_id: str,
    arm_name: str,
    dr_name: str,
    dr_idx: int,
    meta_obs: np.ndarray,
    closed: ClosedConflictInstance,
) -> float | None:
    arm_idx = bandit_arm_names.index(arm_name)
    reward = _bandit_reward_from_episode(
        dr_name,
        team_utility_mean=closed.team_util_norm,
        own_p_mean=closed.p_norm,
    )
    nested_bandits[agent_id][dr_name].update(arm_idx, reward)
    dr_policies[agent_id].store(meta_obs, dr_idx, reward)
    return dr_policies[agent_id].train_step()


def _handle_closed_conflict_instance(
    *,
    closed: ClosedConflictInstance,
    episode_own_p_samples: list[float],
    episode_own_c_samples: list[float],
    episode_team_utility_samples: list[float],
    episode_own_p_by_agent: dict[str, list[float]],
    episode_arm_stats: dict[str, dict[str, dict[str, list[float]]]] | None,
    agent_id: str,
    arm_name: str | None,
    bandits: dict[str, UCB1Bandit] | None = None,
    bandit_arm_names: tuple[str, ...] | None = None,
    bandit_reward_model: str | None = None,
    bandit_credit_mode: str | None = None,
    training_arm_samples: dict[str, dict[str, list[float]]] | None = None,
):
    _record_closed_conflict_instance(
        closed=closed,
        episode_own_p_samples=episode_own_p_samples,
        episode_own_c_samples=episode_own_c_samples,
        episode_team_utility_samples=episode_team_utility_samples,
        episode_own_p_by_agent=episode_own_p_by_agent,
        episode_arm_stats=episode_arm_stats,
        agent_id=agent_id,
        arm_name=arm_name,
    )
    if (
        training_arm_samples is not None
        and bandit_reward_model is not None
        and arm_name is not None
    ):
        reward = _bandit_reward_from_episode(
            bandit_reward_model,
            team_utility_mean=closed.team_util_norm,
            own_p_mean=closed.p_norm,
        )
        delta = _bandit_dr_delta_raw(
            bandit_reward_model,
            team_utility_mean=closed.team_util_norm,
            own_p_mean=closed.p_norm,
        )
        training_arm_samples[arm_name]["reward"].append(reward)
        training_arm_samples[arm_name]["delta"].append(delta)
        training_arm_samples[arm_name]["own_p"].append(closed.p_norm)
        training_arm_samples[arm_name]["team_util"].append(closed.team_util_norm)
    if (
        bandits is not None
        and bandit_arm_names is not None
        and bandit_reward_model is not None
        and bandit_credit_mode == "instance_credited"
        and arm_name is not None
    ):
        _credit_bandit_conflict_instance(
            bandits=bandits,
            bandit_arm_names=bandit_arm_names,
            bandit_reward_model=bandit_reward_model,
            agent_id=agent_id,
            arm_name=arm_name,
            closed=closed,
        )


def _apply_bandit_episode_updates(
    *,
    bandits: dict[str, UCB1Bandit],
    bandit_arm_names: tuple[str, ...],
    bandit_reward_model: str,
    bandit_credit_mode: str,
    episode_arm_stats: dict[str, dict[str, dict[str, list[float]]]],
    conflict_trackers_by_env: list[dict[str, ConflictInstanceTracker]],
    agent_ids: list[str],
):
    team_utility_mean_episode = episode_mean_team_util_norm_all_envs(conflict_trackers_by_env, agent_ids)
    if bandit_credit_mode == "episode_shared":
        own_p_mean_episode = episode_mean_own_p_norm_all_envs(conflict_trackers_by_env, agent_ids)
        reward_for_update = _bandit_reward_from_episode(
            bandit_reward_model,
            team_utility_mean=team_utility_mean_episode,
            own_p_mean=own_p_mean_episode,
        )
        for agent_id in agent_ids:
            for arm_idx, arm_name in enumerate(bandit_arm_names):
                if len(episode_arm_stats[agent_id][arm_name]["own_p"]) > 0:
                    bandits[agent_id].update(arm_idx, reward_for_update)
        return

    for agent_id in agent_ids:
        own_p_episode = float(
            np.mean([conflict_trackers_by_env[env_idx][agent_id].mean_closed_p_norm() for env_idx in range(len(conflict_trackers_by_env))])
        )
        for arm_idx, arm_name in enumerate(bandit_arm_names):
            arm_samples = episode_arm_stats[agent_id][arm_name]
            if len(arm_samples["own_p"]) == 0:
                continue
            own_p_k = _mean_arm_stat(arm_samples["own_p"])
            if bandit_credit_mode == "per_arm_credited":
                team_k = _mean_arm_stat(arm_samples["team_util"])
                reward_k = _bandit_reward_from_episode(
                    bandit_reward_model,
                    team_utility_mean=team_k,
                    own_p_mean=own_p_k,
                )
            elif bandit_credit_mode == "arm_relative_my_role":
                delta = _bandit_dr_delta_raw(
                    bandit_reward_model,
                    team_utility_mean=team_utility_mean_episode,
                    own_p_mean=own_p_episode,
                )
                if bandit_reward_model.startswith("neutral_"):
                    delta += float(own_p_episode - own_p_k)
                reward_k = _dr_softplus(delta)
            else:
                raise ValueError(f"Unsupported bandit_credit_mode in episode update: {bandit_credit_mode}")
            bandits[agent_id].update(arm_idx, reward_k)


def _expert_actions_for_requests(
    expert: FrozenTaskExpert,
    obs_list: list[dict[str, np.ndarray]],
    requests: list[tuple[int, str]],
) -> dict[tuple[int, str], int]:
    if len(requests) == 0:
        return {}
    obs_batch = np.array([obs_list[env_idx][agent_id] for env_idx, agent_id in requests], dtype=np.float32)
    action_batch = expert.select_actions_batch(obs_batch)
    return {(env_idx, agent_id): int(action_batch[idx]) for idx, (env_idx, agent_id) in enumerate(requests)}


def _run_baseline(args):
    print(f"Swarm RL device: {DEVICE}")
    n_agents = args.n_agents
    num_envs = args.num_envs
    grid_size = args.grid_size
    num_food = args.num_food
    local_grid_size = args.local_grid_size
    n_episodes = args.episodes
    max_steps = args.max_steps_per_episode
    if num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.expert_checkpoint is None:
        raise ValueError("--expert_checkpoint is required for baseline modes")

    suppress_collision = args.baseline_mode == "collision_free"
    envs = [
        RationalSwarmForagingEnv(
            n_agents=n_agents,
            grid_size=grid_size,
            num_food=num_food,
            local_grid_size=local_grid_size,
            render_mode=None,
            suppress_agent_collision=suppress_collision,
        )
        for _ in range(num_envs)
    ]
    env0 = envs[0]
    expert = FrozenTaskExpert(obs_dim=OBS_DIM, checkpoint_path=args.expert_checkpoint, n_actions=N_ACTIONS_FULL, hidden_dim=64)
    bandit_arm_names: tuple[str, ...] = tuple(CONFLICT_ACTION_NAMES)
    bandits = {}
    dr_policies: dict[str, DRUCBPolicyLearner] = {}
    if args.baseline_mode == "bandit_ucb1":
        bandit_arm_names = baseline_bandit_arms_tuple(args.bandit_conflict_arms)
        bandits = {
            agent_id: UCB1Bandit(n_arms=len(bandit_arm_names)) for agent_id in env0.possible_agents
        }
    elif args.baseline_mode == "dr_ucb_mixture":
        if args.bandit_credit_mode != "instance_credited":
            raise ValueError("dr_ucb_mixture only supports bandit_credit_mode=instance_credited")
        bandit_arm_names = baseline_bandit_arms_tuple(args.bandit_conflict_arms)
        bandits = {
            agent_id: {
                dr_name: UCB1Bandit(n_arms=len(bandit_arm_names))
                for dr_name in BANDIT_REWARD_MODEL_NAMES
            }
            for agent_id in env0.possible_agents
        }
        dr_policies = {
            agent_id: DRUCBPolicyLearner(
                obs_dim=OBS_DIM,
                n_dr_experts=len(BANDIT_REWARD_MODEL_NAMES),
                lr=args.dr_meta_lr,
                batch_size=args.dr_meta_batch_size,
            )
            for agent_id in env0.possible_agents
        }
    arm_counts = {name: 0 for name in CONFLICT_ACTION_NAMES}
    training_arm_samples = (
        _new_training_arm_samples(bandit_arm_names)
        if args.baseline_mode == "bandit_ucb1"
        else None
    )

    episode_metrics = []
    episode_env_rewards = []
    episode_env_rewards_raw = []
    for episode in range(n_episodes):
        obs_list = []
        for env_idx in range(num_envs):
            obs_i, _ = envs[env_idx].reset(seed=episode * num_envs + env_idx)
            obs_list.append(obs_i)

        macro_queues = [{agent_id: [] for agent_id in env0.possible_agents} for _ in range(num_envs)]
        prev_conflict_flags = [{agent_id: False for agent_id in env0.possible_agents} for _ in range(num_envs)]
        headings = [{agent_id: 0 for agent_id in env0.possible_agents} for _ in range(num_envs)]
        token_remaining = [{agent_id: 0 for agent_id in env0.possible_agents} for _ in range(num_envs)]
        episode_own_p_samples = []
        episode_own_c_samples = []
        episode_conflict_samples = []
        episode_team_utility_samples = []
        episode_own_p_by_agent = {agent_id: [] for agent_id in env0.possible_agents}
        episode_arm_stats = _new_episode_arm_stats(env0.possible_agents, bandit_arm_names)
        conflict_trackers = [
            new_conflict_trackers(env0.possible_agents, args.max_conflict_steps, args.max_conflict_steps_type)
            for _ in range(num_envs)
        ]
        active_conflict_arm = [
            {agent_id: None for agent_id in env0.possible_agents} for _ in range(num_envs)
        ]
        active_dr_model = [
            {agent_id: None for agent_id in env0.possible_agents} for _ in range(num_envs)
        ]
        active_dr_idx = [
            {agent_id: None for agent_id in env0.possible_agents} for _ in range(num_envs)
        ]
        active_meta_obs = [
            {agent_id: None for agent_id in env0.possible_agents} for _ in range(num_envs)
        ]
        episode_dr_meta_losses: list[float] = []
        episode_env_reward_raw = 0.0
        step_count = 0
        episode_arm_counts = {name: 0 for name in CONFLICT_ACTION_NAMES}
        while step_count < max_steps:
            env_actions = [{} for _ in range(num_envs)]
            selected_is_c = [{agent_id: False for agent_id in env0.possible_agents} for _ in range(num_envs)]
            prev_positions_list = [{a: list(envs[e].agent_positions[a]) for a in env0.possible_agents} for e in range(num_envs)]
            expert_requests: list[tuple[int, str]] = []
            for env_idx in range(num_envs):
                for agent_id in env0.possible_agents:
                    if len(macro_queues[env_idx][agent_id]) > 0:
                        continue
                    if not prev_conflict_flags[env_idx][agent_id]:
                        expert_requests.append((env_idx, agent_id))
                        continue
                    if args.baseline_mode == "handshake":
                        continue
                    expert_requests.append((env_idx, agent_id))
            expert_actions = _expert_actions_for_requests(expert, obs_list, expert_requests)
            for env_idx in range(num_envs):
                handshake_selected = set()
                reserve_selected = set()
                reserve_escape_selected = set()
                priority_selected = set()
                pass_food_selected = set()
                freeze_selected = set()
                for agent_id in env0.possible_agents:
                    if len(macro_queues[env_idx][agent_id]) > 0:
                        low_action = macro_queues[env_idx][agent_id].pop(0)
                        selected_is_c[env_idx][agent_id] = True
                        env_actions[env_idx][agent_id] = int(low_action)
                        if args.baseline_mode in ("bandit_ucb1", "dr_ucb_mixture") and active_conflict_arm[env_idx][agent_id] is None:
                            raise ValueError("macro queue active without bandit conflict arm")
                        continue
                    if not prev_conflict_flags[env_idx][agent_id]:
                        if token_remaining[env_idx][agent_id] > 0:
                            token_remaining[env_idx][agent_id] -= 1
                        expert_action = expert_actions[(env_idx, agent_id)]
                        env_actions[env_idx][agent_id] = expert_action
                        selected_is_c[env_idx][agent_id] = False
                        active_conflict_arm[env_idx][agent_id] = None
                        continue
                    if token_remaining[env_idx][agent_id] > 0:
                        token_remaining[env_idx][agent_id] -= 1
                        expert_action = expert_actions[(env_idx, agent_id)]
                        env_actions[env_idx][agent_id] = expert_action
                        selected_is_c[env_idx][agent_id] = True
                        continue

                    if args.baseline_mode == "random_conflict":
                        action_name = _arm_to_name(int(np.random.randint(0, len(CONFLICT_ACTION_NAMES))))
                    elif args.baseline_mode == "fixed_conflict":
                        action_name = args.fixed_conflict_action
                    elif args.baseline_mode == "bandit_ucb1":
                        selected_arm_idx = int(bandits[agent_id].select_arm())
                        action_name = bandit_arm_names[selected_arm_idx]
                        active_conflict_arm[env_idx][agent_id] = action_name
                    elif args.baseline_mode == "dr_ucb_mixture":
                        dr_idx, _ = dr_policies[agent_id].select_dr(obs_list[env_idx][agent_id])
                        dr_name = BANDIT_REWARD_MODEL_NAMES[dr_idx]
                        selected_arm_idx = int(bandits[agent_id][dr_name].select_arm())
                        action_name = bandit_arm_names[selected_arm_idx]
                        active_dr_model[env_idx][agent_id] = dr_name
                        active_dr_idx[env_idx][agent_id] = dr_idx
                        active_meta_obs[env_idx][agent_id] = obs_list[env_idx][agent_id].copy()
                        active_conflict_arm[env_idx][agent_id] = action_name
                    else:
                        raise ValueError(f"Unsupported baseline_mode: {args.baseline_mode}")
                    action_name = _canonical_conflict_action_name(action_name)
                    if args.baseline_mode in ("bandit_ucb1", "dr_ucb_mixture"):
                        active_conflict_arm[env_idx][agent_id] = action_name
                    arm_counts[action_name] += 1
                    episode_arm_counts[action_name] += 1
                    selected_is_c[env_idx][agent_id] = True
                    if action_name == "handshake":
                        handshake_selected.add(agent_id)
                        continue
                    if action_name == "reserve_parity":
                        reserve_selected.add(agent_id)
                        continue
                    if action_name == "reserve_parity_escape":
                        reserve_escape_selected.add(agent_id)
                        continue
                    if action_name == "priority_swap_n3":
                        priority_selected.add(agent_id)
                        continue
                    if action_name == "pass_food_n3":
                        pass_food_selected.add(agent_id)
                        continue
                    if action_name == "freeze_tag":
                        freeze_selected.add(agent_id)
                        continue
                    expert_action = expert_actions[(env_idx, agent_id)]
                    seq = _macro_action_sequence(action_name, expert_action, obs_list[env_idx][agent_id])
                    macro_queues[env_idx][agent_id] = seq[1:]
                    env_actions[env_idx][agent_id] = int(seq[0])
                if len(handshake_selected) > 0:
                    resolve_handshake_step(
                        handshake_selected=handshake_selected,
                        low_actions=env_actions[env_idx],
                        macro_queues=macro_queues[env_idx],
                        headings=headings[env_idx],
                        positions=prev_positions_list[env_idx],
                    )
                if len(freeze_selected) > 0:
                    resolve_freeze_tag_step(
                        freeze_selected=freeze_selected,
                        env=envs[env_idx],
                        expert_actions_subset={aid: expert_actions[(env_idx, aid)] for aid in freeze_selected},
                        low_actions=env_actions[env_idx],
                        macro_queues=macro_queues[env_idx],
                        positions=prev_positions_list[env_idx],
                    )
                if len(reserve_selected) > 0:
                    resolve_reserve_parity_step(
                        reserve_selected=reserve_selected,
                        low_actions=env_actions[env_idx],
                        macro_queues=macro_queues[env_idx],
                        headings=headings[env_idx],
                        positions=prev_positions_list[env_idx],
                        grid_size=env0.grid_size,
                    )
                if len(reserve_escape_selected) > 0:
                    resolve_reserve_parity_escape_step(
                        reserve_selected=reserve_escape_selected,
                        low_actions=env_actions[env_idx],
                        macro_queues=macro_queues[env_idx],
                        headings=headings[env_idx],
                        positions=prev_positions_list[env_idx],
                        grid_size=env0.grid_size,
                    )
                if len(priority_selected) > 0:
                    resolve_priority_swap_selection_step(
                        priority_selected=priority_selected,
                        low_actions=env_actions[env_idx],
                        macro_queues=macro_queues[env_idx],
                        expert_actions_by_agent={a: expert_actions[(env_idx, a)] for a in priority_selected},
                        positions=prev_positions_list[env_idx],
                        token_remaining=token_remaining[env_idx],
                        grid_size=env0.grid_size,
                    )
                if len(pass_food_selected) > 0:
                    resolve_pass_food_n3_step(
                        pass_selected=pass_food_selected,
                        low_actions=env_actions[env_idx],
                        macro_queues=macro_queues[env_idx],
                        positions=prev_positions_list[env_idx],
                        agent_holding_food=envs[env_idx].agent_holding_food,
                        base_position=tuple(envs[env_idx].base_position),
                        obs_by_agent=obs_list[env_idx],
                    )
                resolve_priority_swap_token_step(
                    env_actions=env_actions[env_idx],
                    macro_queues=macro_queues[env_idx],
                    prev_positions=prev_positions_list[env_idx],
                    prev_conflict_flags=prev_conflict_flags[env_idx],
                    token_remaining=token_remaining[env_idx],
                    grid_size=env0.grid_size,
                )

            next_obs_list = []
            infos_list = []
            for env_idx in range(num_envs):
                next_obs, _rewards, _terminations, _truncations, infos = envs[env_idx].step(env_actions[env_idx])
                next_obs_list.append(next_obs)
                infos_list.append(infos)
                episode_env_reward_raw += infos[env0.possible_agents[0]]["env_reward"]
                team_utility = step_team_utility_mean(infos, env0.possible_agents)
                episode_team_utility_samples.append(float(team_utility))
                new_positions = {a: list(envs[env_idx].agent_positions[a]) for a in env0.possible_agents}
                for agent_id in env0.possible_agents:
                    is_c = float(selected_is_c[env_idx][agent_id])
                    if args.baseline_mode == "collision_free":
                        episode_conflict_samples.append(0.0)
                        prev_conflict_flags[env_idx][agent_id] = False
                    else:
                        has_conflict = float(infos[agent_id]["n_local"] > 1.0)
                        episode_conflict_samples.append(has_conflict)
                        prev_conflict_flags[env_idx][agent_id] = bool(has_conflict)
                    tracker = conflict_trackers[env_idx][agent_id]
                    closed = tracker.step(
                        p_t=float(infos[agent_id]["p_t"]),
                        c_t=float(infos[agent_id]["c_t"]),
                        team_util=float(team_utility),
                        n_local=float(infos[agent_id]["n_local"]),
                        new_conflict_event=float(infos[agent_id]["new_conflict_event"]),
                        agent_moved=_agent_moved(prev_positions_list[env_idx][agent_id], new_positions[agent_id]),
                    )
                    arm_name = active_conflict_arm[env_idx][agent_id]
                    if closed is not None:
                        _handle_closed_conflict_instance(
                            closed=closed,
                            episode_own_p_samples=episode_own_p_samples,
                            episode_own_c_samples=episode_own_c_samples,
                            episode_team_utility_samples=episode_team_utility_samples,
                            episode_own_p_by_agent=episode_own_p_by_agent,
                            episode_arm_stats=episode_arm_stats if args.baseline_mode in ("bandit_ucb1", "dr_ucb_mixture") else None,
                            agent_id=agent_id,
                            arm_name=arm_name,
                            bandits=bandits if args.baseline_mode == "bandit_ucb1" else None,
                            bandit_arm_names=bandit_arm_names if args.baseline_mode == "bandit_ucb1" else None,
                            bandit_reward_model=args.bandit_reward_model if args.baseline_mode == "bandit_ucb1" else None,
                            bandit_credit_mode=args.bandit_credit_mode if args.baseline_mode == "bandit_ucb1" else None,
                            training_arm_samples=training_arm_samples,
                        )
                        if args.baseline_mode == "dr_ucb_mixture" and arm_name is not None:
                            meta_loss = _credit_dr_ucb_mixture_instance(
                                nested_bandits=bandits,
                                dr_policies=dr_policies,
                                bandit_arm_names=bandit_arm_names,
                                agent_id=agent_id,
                                arm_name=arm_name,
                                dr_name=active_dr_model[env_idx][agent_id],
                                dr_idx=active_dr_idx[env_idx][agent_id],
                                meta_obs=active_meta_obs[env_idx][agent_id],
                                closed=closed,
                            )
                            if meta_loss is not None:
                                episode_dr_meta_losses.append(meta_loss)
                    if args.baseline_mode == "bandit_ucb1":
                        if arm_name is not None and tracker.active and args.bandit_credit_mode == "step_level":
                            arm_idx = bandit_arm_names.index(arm_name)
                            reward_step = _bandit_reward_from_episode(
                                args.bandit_reward_model,
                                team_utility_mean=tracker.running_team_util_norm(),
                                own_p_mean=tracker.running_p_norm(),
                            )
                            bandits[agent_id].update(arm_idx, reward_step)
                    if args.baseline_mode in ("bandit_ucb1", "dr_ucb_mixture"):
                        if len(macro_queues[env_idx][agent_id]) == 0:
                            active_conflict_arm[env_idx][agent_id] = None
                            if args.baseline_mode == "dr_ucb_mixture":
                                active_dr_model[env_idx][agent_id] = None
                                active_dr_idx[env_idx][agent_id] = None
                                active_meta_obs[env_idx][agent_id] = None
                update_headings_from_step(headings[env_idx], prev_positions_list[env_idx], new_positions, env_actions[env_idx])
            obs_list = next_obs_list
            step_count += 1

        episode_env_reward = float(episode_env_reward_raw / num_envs)
        episode_env_rewards.append(episode_env_reward)
        episode_env_rewards_raw.append(float(episode_env_reward_raw))
        eps = 0.0

        for env_idx in range(num_envs):
            for agent_id in env0.possible_agents:
                closed = conflict_trackers[env_idx][agent_id].finalize_episode()
                if closed is not None:
                    _handle_closed_conflict_instance(
                        closed=closed,
                        episode_own_p_samples=episode_own_p_samples,
                        episode_own_c_samples=episode_own_c_samples,
                        episode_team_utility_samples=episode_team_utility_samples,
                        episode_own_p_by_agent=episode_own_p_by_agent,
                        episode_arm_stats=episode_arm_stats if args.baseline_mode in ("bandit_ucb1", "dr_ucb_mixture") else None,
                        agent_id=agent_id,
                        arm_name=active_conflict_arm[env_idx][agent_id],
                        bandits=bandits if args.baseline_mode == "bandit_ucb1" else None,
                        bandit_arm_names=bandit_arm_names if args.baseline_mode == "bandit_ucb1" else None,
                        bandit_reward_model=args.bandit_reward_model if args.baseline_mode == "bandit_ucb1" else None,
                        bandit_credit_mode=args.bandit_credit_mode if args.baseline_mode == "bandit_ucb1" else None,
                        training_arm_samples=training_arm_samples,
                    )
                    if args.baseline_mode == "dr_ucb_mixture" and active_conflict_arm[env_idx][agent_id] is not None:
                        meta_loss = _credit_dr_ucb_mixture_instance(
                            nested_bandits=bandits,
                            dr_policies=dr_policies,
                            bandit_arm_names=bandit_arm_names,
                            agent_id=agent_id,
                            arm_name=active_conflict_arm[env_idx][agent_id],
                            dr_name=active_dr_model[env_idx][agent_id],
                            dr_idx=active_dr_idx[env_idx][agent_id],
                            meta_obs=active_meta_obs[env_idx][agent_id],
                            closed=closed,
                        )
                        if meta_loss is not None:
                            episode_dr_meta_losses.append(meta_loss)

        if args.baseline_mode == "bandit_ucb1" and args.bandit_credit_mode not in ("step_level", "instance_credited"):
            _apply_bandit_episode_updates(
                bandits=bandits,
                bandit_arm_names=bandit_arm_names,
                bandit_reward_model=args.bandit_reward_model,
                bandit_credit_mode=args.bandit_credit_mode,
                episode_arm_stats=episode_arm_stats,
                conflict_trackers_by_env=conflict_trackers,
                agent_ids=env0.possible_agents,
            )

        bandit_method_suffix = (
            f"{args.bandit_reward_model}_{args.bandit_credit_mode}"
            if args.baseline_mode == "bandit_ucb1"
            else ""
        )
        if args.baseline_mode == "dr_ucb_mixture":
            method_label = f"dr_ucb_mixture_{args.bandit_credit_mode}"
        elif args.baseline_mode == "bandit_ucb1":
            method_label = f"bandit_ucb1_{bandit_method_suffix}"
        else:
            method_label = args.baseline_mode
        metric_row = {
            "episode": episode + 1,
            "method": method_label,
            "bandit_credit_mode": args.bandit_credit_mode if args.baseline_mode in ("bandit_ucb1", "dr_ucb_mixture") else "",
            "n_agents": float(n_agents),
            "avg_env_reward_last_50": float(np.mean(episode_env_rewards[-50:])),
            "episode_env_reward": float(episode_env_reward),
            "episode_env_reward_raw": float(episode_env_reward_raw),
            "avg_p_time_fraction": float(np.mean(episode_own_p_samples)),
            "avg_p_time_percent": float(100.0 * np.mean(episode_own_p_samples)),
            "avg_c_time_fraction": float(np.mean(episode_own_c_samples)),
            "avg_c_time_percent": float(100.0 * np.mean(episode_own_c_samples)),
            "avg_conflict_fraction": float(np.mean(episode_conflict_samples)),
            "avg_conflict_percent": float(100.0 * np.mean(episode_conflict_samples)),
            "epsilon": float(eps),
            "policy_loss_mean": np.nan,
            "critic_loss_mean": np.nan,
            "dr_loss_mean": np.nan,
            "dr_meta_loss_mean": float(np.mean(episode_dr_meta_losses)) if len(episode_dr_meta_losses) > 0 else np.nan,
            "dr_mixed_reward_mean": np.nan,
            "u_solver_mean": np.nan,
            "u_neutral_mean": np.nan,
            "u_causer_mean": np.nan,
            "v_c_mean": np.nan,
            "v_p_mean": np.nan,
            "v_mean_mean": np.nan,
            "my_mean_mean": np.nan,
            "arm_wait3_count": float(arm_counts["wait3"]),
            "arm_backward3_count": float(arm_counts["backward3"]),
            "arm_randomwalk3_count": float(arm_counts["randomwalk3"]),
            "arm_wait2_forward1_count": float(arm_counts["wait2_forward1"]),
            "arm_move_clear_count": float(arm_counts["move_clear"]),
            "arm_handshake_count": float(arm_counts["handshake"]),
            "arm_reserve_parity_count": float(arm_counts["reserve_parity"]),
            "arm_reserve_parity_escape_count": float(arm_counts["reserve_parity_escape"]),
            "arm_priority_swap_n3_count": float(arm_counts["priority_swap_n3"]),
            "arm_pass_food_n3_count": float(arm_counts["pass_food_n3"]),
            "arm_freeze_tag_count": float(arm_counts["freeze_tag"]),
        }
        if args.baseline_mode == "bandit_ucb1":
            probs = np.mean(np.stack([bandits[a].arm_probabilities() for a in env0.possible_agents], axis=0), axis=0)
            values = np.mean(np.stack([bandits[a].values for a in env0.possible_agents], axis=0), axis=0)
            for name in CONFLICT_ACTION_NAMES:
                metric_row[f"arm_{name}_prob"] = np.nan
                metric_row[f"arm_{name}_value"] = np.nan
            for arm_idx, name in enumerate(bandit_arm_names):
                metric_row[f"arm_{name}_prob"] = float(probs[arm_idx])
                metric_row[f"arm_{name}_value"] = float(values[arm_idx])
            metric_row.update(
                _bandit_arm_metric_columns(
                    bandit_arm_names=bandit_arm_names,
                    bandits=bandits,
                    agent_ids=env0.possible_agents,
                    training_arm_samples=training_arm_samples,
                    episode_arm_counts=episode_arm_counts,
                    arm_counts=arm_counts,
                )
            )
        else:
            metric_row["arm_wait3_prob"] = np.nan
            metric_row["arm_backward3_prob"] = np.nan
            metric_row["arm_randomwalk3_prob"] = np.nan
            metric_row["arm_wait2_forward1_prob"] = np.nan
            metric_row["arm_move_clear_prob"] = np.nan
            metric_row["arm_handshake_prob"] = np.nan
            metric_row["arm_reserve_parity_prob"] = np.nan
            metric_row["arm_reserve_parity_escape_prob"] = np.nan
            metric_row["arm_priority_swap_n3_prob"] = np.nan
            metric_row["arm_pass_food_n3_prob"] = np.nan
            metric_row["arm_freeze_tag_prob"] = np.nan
            for name in CONFLICT_ACTION_NAMES:
                metric_row[f"arm_{name}_value"] = np.nan
            for name in CONFLICT_ACTION_NAMES:
                for suffix in (
                    "pull_count",
                    "ep_sel_frac",
                    "cum_sel_frac",
                    "reward_mean",
                    "reward_var",
                    "delta_mean",
                    "delta_var",
                    "own_p_mean",
                    "team_util_mean",
                    "team_util_var",
                ):
                    metric_row[f"arm_{name}_{suffix}"] = np.nan
        episode_metrics.append(metric_row)

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_env_rewards[-50:])
            print(f"Episode {episode + 1}/{n_episodes} | Avg env reward (last 50): {avg:.2f} | baseline: {args.baseline_mode}")

    for env_i in envs:
        env_i.close()
    print(
        "Training complete. "
        f"Final avg env reward per env: {np.mean(episode_env_rewards[-50:]):.2f} | "
        f"raw total across envs: {np.mean(episode_env_rewards_raw[-50:]):.2f}"
    )

    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", newline="", encoding="utf-8") as csv_file:
        if len(episode_metrics) == 0:
            raise ValueError("episode_metrics is unexpectedly empty")
        writer = csv.DictWriter(csv_file, fieldnames=list(episode_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(episode_metrics)
    print(f"Saved training metrics to {metrics_path}")


def run_training(args):
    if args.baseline_mode != "none":
        _run_baseline(args)
        return
    n_agents = args.n_agents
    num_envs = args.num_envs
    grid_size = args.grid_size
    num_food = args.num_food
    local_grid_size = args.local_grid_size
    n_episodes = args.episodes
    max_steps = args.max_steps_per_episode
    mode = args.mode
    if num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if mode == "task_avoid" and args.expert_checkpoint is None:
        raise ValueError("--expert_checkpoint is required when --mode task_avoid")
    if mode == "task_avoid" and args.load_weights is not None:
        raise ValueError("--load_weights is for 5-action full mode; use it with --mode full")
    if args.train_every <= 0:
        raise ValueError("--train_every must be positive")

    print(f"Swarm RL device: {DEVICE}")

    envs = [
        RationalSwarmForagingEnv(
            n_agents=n_agents,
            grid_size=grid_size,
            num_food=num_food,
            local_grid_size=local_grid_size,
            render_mode=None,
        )
        for _ in range(num_envs)
    ]
    env0 = envs[0]
    if mode == "task_avoid" and len(TASK_AVOID_ENABLED_ACTION_IDS) == 0:
        raise ValueError("TASK_AVOID_ENABLED_ACTION_IDS cannot be empty")
    learned_n_actions = N_ACTIONS_FULL if mode == "full" else N_ACTIONS_TASK_AVOID
    expert = None
    if mode == "task_avoid":
        expert = FrozenTaskExpert(obs_dim=OBS_DIM, checkpoint_path=args.expert_checkpoint, n_actions=N_ACTIONS_FULL, hidden_dim=64)

    policies = {
        agent_id: DQNAgent(
            obs_dim=OBS_DIM,
            n_actions=learned_n_actions,
            lr=args.alpha,
            gamma=args.gamma,
            epsilon_start=1.0,
            epsilon_end=args.epsilon_min,
            epsilon_decay_steps=args.epsilon_decay,
            buffer_size=args.buffer_size,
            batch_size=64,
            target_update_freq=args.target_update_freq,
        )
        for agent_id in env0.possible_agents
    }
    if args.load_weights is not None:
        state_dict = torch.load(args.load_weights, map_location=DEVICE)
        for agent_id in env0.possible_agents:
            policies[agent_id].q_net.load_state_dict(state_dict)
            policies[agent_id].target_net.load_state_dict(state_dict)
        print(f"Initialized all {n_agents} agents from {args.load_weights}")

    critics = {
        agent_id: PCCriticLearner(
            obs_dim=OBS_DIM,
            lr=args.critic_lr,
            gamma=args.critic_gamma,
            buffer_size=args.critic_buffer_size,
            batch_size=64,
            target_update_freq=args.critic_target_update_freq,
        )
        for agent_id in env0.possible_agents
    }
    dr_learners = {}
    if args.reward_mode == "scenario_mixture":
        dr_learners = {
            agent_id: DRScenarioMixtureLearner(
                obs_dim=OBS_DIM,
                lr=args.dr_lr,
                hidden_dim=64,
                entropy_coef=args.dr_entropy_coef,
                full_duration=float(FULL_DURATION_NORM),
                my_mean_mode=args.dr_my_mean_mode,
            )
            for agent_id in env0.possible_agents
        }

    episode_env_rewards = []
    episode_env_rewards_raw = []
    episode_metrics = []
    for episode in range(n_episodes):
        obs_list = []
        for env_idx in range(num_envs):
            obs_i, _ = envs[env_idx].reset(seed=episode * num_envs + env_idx)
            obs_list.append(obs_i)

        episode_train_inputs = {agent_id: {"obs": [], "team_utility": [], "own_p": [], "own_c": []} for agent_id in env0.possible_agents}
        episode_own_p_samples = []
        episode_own_c_samples = []
        episode_conflict_samples = []
        episode_team_utility_samples = []
        conflict_trackers = [
            new_conflict_trackers(env0.possible_agents, args.max_conflict_steps, args.max_conflict_steps_type)
            for _ in range(num_envs)
        ]
        episode_own_p_by_agent = {agent_id: [] for agent_id in env0.possible_agents}
        task_macro_queues = [{agent_id: [] for agent_id in env0.possible_agents} for _ in range(num_envs)]
        task_macro_current_hi = [{agent_id: 0 for agent_id in env0.possible_agents} for _ in range(num_envs)]
        task_headings = [{agent_id: 0 for agent_id in env0.possible_agents} for _ in range(num_envs)]

        episode_env_reward_raw = 0.0
        step_count = 0
        while step_count < max_steps:
            prev_positions_list = [{a: list(envs[e].agent_positions[a]) for a in env0.possible_agents} for e in range(num_envs)]
            if mode == "task_avoid":
                actions_hi_list = [{} for _ in range(num_envs)]
                actions_low_list = [{} for _ in range(num_envs)]
                expert_requests: list[tuple[int, str]] = []
                for agent_id in env0.possible_agents:
                    policy = policies[agent_id]
                    obs_batch = np.array([obs_list[e][agent_id] for e in range(num_envs)], dtype=np.float32)
                    hi_batch = policy.select_actions_batch(obs_batch)
                    for env_idx in range(num_envs):
                        expert_needed = False
                        if len(task_macro_queues[env_idx][agent_id]) > 0:
                            a_hi_idx = task_macro_current_hi[env_idx][agent_id]
                            a_hi = TASK_AVOID_ENABLED_ACTION_IDS[a_hi_idx]
                            low_action = task_macro_queues[env_idx][agent_id].pop(0)
                        else:
                            a_hi_idx = int(hi_batch[env_idx])
                            task_macro_current_hi[env_idx][agent_id] = a_hi_idx
                            a_hi = TASK_AVOID_ENABLED_ACTION_IDS[a_hi_idx]
                            if a_hi == TASK_POLICY_ACTION:
                                expert_needed = True
                                low_action = 0
                            elif a_hi == NO_OP_3_TURNS_ACTION:
                                task_macro_queues[env_idx][agent_id] = [4, 4]
                                low_action = 4
                            elif a_hi == MOVE_BACKWARDS_2_ACTION:
                                expert_needed = True
                                low_action = 0
                            elif a_hi == RANDOM_WALK_3_ACTION:
                                rw = random_walk_actions(3)
                                low_action = rw[0]
                                task_macro_queues[env_idx][agent_id] = rw[1:]
                            elif a_hi == WAIT2_THEN_FORWARD1_ACTION:
                                expert_needed = True
                                low_action = 4
                                task_macro_queues[env_idx][agent_id] = [4, 0]
                                low_action = 4
                            elif a_hi == MOVE_CLEAR_ACTION:
                                low_action = move_clear_action(obs_list[env_idx][agent_id])
                            elif a_hi == HANDSHAKE_ACTION:
                                low_action = 4
                            elif a_hi == RESERVE_PARITY_ACTION:
                                low_action = 4
                            elif a_hi == RESERVE_PARITY_ESCAPE_ACTION:
                                low_action = 4
                            else:
                                raise ValueError(f"Unknown high-level task_avoid action: {a_hi}")
                        if expert_needed:
                            expert_requests.append((env_idx, agent_id))
                        actions_hi_list[env_idx][agent_id] = a_hi_idx
                        actions_low_list[env_idx][agent_id] = low_action
                expert_actions = _expert_actions_for_requests(expert, obs_list, expert_requests)
                for env_idx, agent_id in expert_requests:
                    a_hi_idx = actions_hi_list[env_idx][agent_id]
                    a_hi = TASK_AVOID_ENABLED_ACTION_IDS[a_hi_idx]
                    expert_action = expert_actions[(env_idx, agent_id)]
                    if a_hi == TASK_POLICY_ACTION:
                        actions_low_list[env_idx][agent_id] = expert_action
                    elif a_hi == MOVE_BACKWARDS_2_ACTION:
                        backward_action = invert_move(expert_action)
                        task_macro_queues[env_idx][agent_id] = [backward_action]
                        actions_low_list[env_idx][agent_id] = backward_action
                    elif a_hi == WAIT2_THEN_FORWARD1_ACTION:
                        task_macro_queues[env_idx][agent_id] = [4, expert_action]
                    else:
                        raise ValueError(f"Unexpected expert-backed action id: {a_hi}")
                for env_idx in range(num_envs):
                    handshake_selected = set()
                    reserve_selected = set()
                    reserve_escape_selected = set()
                    for agent_id in env0.possible_agents:
                        a_hi_idx = actions_hi_list[env_idx][agent_id]
                        a_hi = TASK_AVOID_ENABLED_ACTION_IDS[a_hi_idx]
                        if a_hi == HANDSHAKE_ACTION and len(task_macro_queues[env_idx][agent_id]) == 0:
                            handshake_selected.add(agent_id)
                        if a_hi == RESERVE_PARITY_ACTION and len(task_macro_queues[env_idx][agent_id]) == 0:
                            reserve_selected.add(agent_id)
                        if a_hi == RESERVE_PARITY_ESCAPE_ACTION and len(task_macro_queues[env_idx][agent_id]) == 0:
                            reserve_escape_selected.add(agent_id)
                    if len(handshake_selected) > 0:
                        resolve_handshake_step(
                            handshake_selected=handshake_selected,
                            low_actions=actions_low_list[env_idx],
                            macro_queues=task_macro_queues[env_idx],
                            headings=task_headings[env_idx],
                            positions=prev_positions_list[env_idx],
                        )
                    if len(reserve_selected) > 0:
                        resolve_reserve_parity_step(
                            reserve_selected=reserve_selected,
                            low_actions=actions_low_list[env_idx],
                            macro_queues=task_macro_queues[env_idx],
                            headings=task_headings[env_idx],
                            positions=prev_positions_list[env_idx],
                            grid_size=env0.grid_size,
                        )
                    if len(reserve_escape_selected) > 0:
                        resolve_reserve_parity_escape_step(
                            reserve_selected=reserve_escape_selected,
                            low_actions=actions_low_list[env_idx],
                            macro_queues=task_macro_queues[env_idx],
                            headings=task_headings[env_idx],
                            positions=prev_positions_list[env_idx],
                            grid_size=env0.grid_size,
                        )
                actor_actions = actions_hi_list
                env_actions = actions_low_list
            else:
                actions_list = [{} for _ in range(num_envs)]
                for agent_id in env0.possible_agents:
                    policy = policies[agent_id]
                    obs_batch = np.array([obs_list[e][agent_id] for e in range(num_envs)], dtype=np.float32)
                    action_batch = policy.select_actions_batch(obs_batch)
                    for env_idx in range(num_envs):
                        actions_list[env_idx][agent_id] = int(action_batch[env_idx])
                actor_actions = actions_list
                env_actions = actions_list

            next_obs_list = []
            terminations_list = []
            truncations_list = []
            infos_list = []
            for env_idx in range(num_envs):
                next_obs, _rewards, terminations, truncations, infos = envs[env_idx].step(env_actions[env_idx])
                next_obs_list.append(next_obs)
                terminations_list.append(terminations)
                truncations_list.append(truncations)
                infos_list.append(infos)
                episode_env_reward_raw += infos[env0.possible_agents[0]]["env_reward"]

            new_positions_list = [
                {a: list(envs[e].agent_positions[a]) for a in env0.possible_agents} for e in range(num_envs)
            ]
            if mode == "task_avoid":
                for env_idx in range(num_envs):
                    update_headings_from_step(
                        task_headings[env_idx],
                        prev_positions_list[env_idx],
                        new_positions_list[env_idx],
                        env_actions[env_idx],
                    )

            team_utility_batch = np.array(
                [step_team_utility_mean(infos, env0.possible_agents) for infos in infos_list],
                dtype=np.float32,
            )
            for env_idx in range(num_envs):
                team_util = float(team_utility_batch[env_idx])
                for agent_id in env0.possible_agents:
                    agent_info = infos_list[env_idx][agent_id]
                    closed = conflict_trackers[env_idx][agent_id].step(
                        p_t=float(agent_info["p_t"]),
                        c_t=float(agent_info["c_t"]),
                        team_util=team_util,
                        n_local=float(agent_info["n_local"]),
                        new_conflict_event=float(agent_info["new_conflict_event"]),
                        agent_moved=_agent_moved(
                            prev_positions_list[env_idx][agent_id],
                            new_positions_list[env_idx][agent_id],
                        ),
                    )
                    if closed is not None:
                        _record_closed_conflict_instance(
                            closed=closed,
                            episode_own_p_samples=episode_own_p_samples,
                            episode_own_c_samples=episode_own_c_samples,
                            episode_team_utility_samples=episode_team_utility_samples,
                            episode_own_p_by_agent=episode_own_p_by_agent,
                            episode_arm_stats=None,
                            agent_id=agent_id,
                            arm_name=None,
                        )

            truncated = (step_count + 1) >= max_steps
            for agent_id in env0.possible_agents:
                policy = policies[agent_id]
                critic = critics[agent_id]
                obs_batch = np.array([obs_list[e][agent_id] for e in range(num_envs)], dtype=np.float32)
                next_obs_batch = np.array([next_obs_list[e][agent_id] for e in range(num_envs)], dtype=np.float32)
                own_p_batch = np.array(
                    [
                        conflict_trackers[e][agent_id].running_p_norm()
                        if conflict_trackers[e][agent_id].active
                        else 0.0
                        for e in range(num_envs)
                    ],
                    dtype=np.float32,
                )
                own_c_batch = np.array(
                    [
                        conflict_trackers[e][agent_id].running_c_norm()
                        if conflict_trackers[e][agent_id].active
                        else 0.0
                        for e in range(num_envs)
                    ],
                    dtype=np.float32,
                )
                team_utility_norm_batch = np.array(
                    [
                        conflict_trackers[e][agent_id].running_team_util_norm()
                        if conflict_trackers[e][agent_id].active
                        else 0.0
                        for e in range(num_envs)
                    ],
                    dtype=np.float32,
                )
                n_local_batch = np.array([float(infos_list[e][agent_id]["n_local"]) for e in range(num_envs)], dtype=np.float32)
                episode_conflict_samples.extend((n_local_batch > 1.0).astype(np.float32).tolist())

                if args.reward_mode == "mean_DR":
                    p_hat_batch, c_hat_batch = critic.predict_rates_batch(obs_batch)
                    reward_batch = env0.beta * p_hat_batch - env0.alpha * c_hat_batch - (n_local_batch - 1.0) * (env0.alpha + env0.beta) * c_hat_batch
                else:
                    reward_batch, _dr_eval = dr_learners[agent_id].reward_batch(
                        obs_batch=obs_batch,
                        team_utility_batch=team_utility_norm_batch,
                        own_p_batch=own_p_batch,
                        own_c_batch=own_c_batch,
                    )
                    active_envs = [env_idx for env_idx in range(num_envs) if conflict_trackers[env_idx][agent_id].active]
                    episode_train_inputs[agent_id]["obs"].extend(obs_batch[active_envs])
                    episode_train_inputs[agent_id]["team_utility"].extend(team_utility_norm_batch[active_envs].tolist())
                    episode_train_inputs[agent_id]["own_p"].extend(own_p_batch[active_envs].tolist())
                    episode_train_inputs[agent_id]["own_c"].extend(own_c_batch[active_envs].tolist())

                for env_idx in range(num_envs):
                    critic.store(
                        obs_list[env_idx][agent_id],
                        float(infos_list[env_idx][agent_id]["p_t"]),
                        float(infos_list[env_idx][agent_id]["c_t"]),
                        next_obs_list[env_idx][agent_id],
                    )
                    done = terminations_list[env_idx][agent_id] or truncations_list[env_idx][agent_id] or truncated
                    policy.store(obs_list[env_idx][agent_id], int(actor_actions[env_idx][agent_id]), float(reward_batch[env_idx]), next_obs_list[env_idx][agent_id], done)

            obs_list = next_obs_list
            step_count += 1

        for env_idx in range(num_envs):
            for agent_id in env0.possible_agents:
                closed = conflict_trackers[env_idx][agent_id].finalize_episode()
                if closed is not None:
                    _record_closed_conflict_instance(
                        closed=closed,
                        episode_own_p_samples=episode_own_p_samples,
                        episode_own_c_samples=episode_own_c_samples,
                        episode_team_utility_samples=episode_team_utility_samples,
                        episode_own_p_by_agent=episode_own_p_by_agent,
                        episode_arm_stats=None,
                        agent_id=agent_id,
                        arm_name=None,
                    )

        episode_env_reward = float(episode_env_reward_raw / num_envs)
        episode_env_rewards_raw.append(float(episode_env_reward_raw))
        episode_env_rewards.append(episode_env_reward)
        policy_losses = []
        critic_losses = []
        dr_losses = []
        dr_u_solver_vals = []
        dr_u_neutral_vals = []
        dr_u_causer_vals = []
        dr_v_c_vals = []
        dr_v_p_vals = []
        dr_v_mean_vals = []
        dr_mixed_reward_vals = []
        dr_my_mean_vals = []

        for agent_id in env0.possible_agents:
            policy_loss = policies[agent_id].train_step()
            if policy_loss is not None:
                policy_losses.append(float(policy_loss))
            critic_loss = critics[agent_id].train_step()
            if critic_loss is not None:
                critic_losses.append(float(critic_loss))
            if args.reward_mode == "scenario_mixture":
                if len(episode_train_inputs[agent_id]["obs"]) == 0:
                    raise ValueError("Scenario mixture training inputs are unexpectedly empty")
                dr_stats = dr_learners[agent_id].train_step_batch(
                    obs_batch=np.array(episode_train_inputs[agent_id]["obs"], dtype=np.float32),
                    team_utility_batch=np.array(episode_train_inputs[agent_id]["team_utility"], dtype=np.float32),
                    own_p_batch=np.array(episode_train_inputs[agent_id]["own_p"], dtype=np.float32),
                    own_c_batch=np.array(episode_train_inputs[agent_id]["own_c"], dtype=np.float32),
                )
                dr_losses.append(float(dr_stats["loss"]))
                dr_u_solver_vals.append(float(dr_stats["u_solver"]))
                dr_u_neutral_vals.append(float(dr_stats["u_neutral"]))
                dr_u_causer_vals.append(float(dr_stats["u_causer"]))
                dr_v_c_vals.append(float(dr_stats["v_c"]))
                dr_v_p_vals.append(float(dr_stats["v_p"]))
                dr_v_mean_vals.append(float(dr_stats["v_mean"]))
                dr_mixed_reward_vals.append(float(dr_stats["mixed_reward"]))
                dr_my_mean_vals.append(float(dr_stats["my_mean"]))

        for agent_id in env0.possible_agents:
            policies[agent_id].advance_epsilon_episode()

        eps = policies[env0.possible_agents[0]].epsilon()
        metric_row = {
            "episode": episode + 1,
            "method": "dqn_dr" if args.reward_mode == "scenario_mixture" else "dqn_mean_dr",
            "n_agents": float(n_agents),
            "avg_env_reward_last_50": float(np.mean(episode_env_rewards[-50:])),
            "episode_env_reward": float(episode_env_reward),
            "episode_env_reward_raw": float(episode_env_reward_raw),
            "avg_p_time_fraction": float(np.mean(episode_own_p_samples)),
            "avg_p_time_percent": float(100.0 * np.mean(episode_own_p_samples)),
            "avg_c_time_fraction": float(np.mean(episode_own_c_samples)),
            "avg_c_time_percent": float(100.0 * np.mean(episode_own_c_samples)),
            "avg_conflict_fraction": float(np.mean(episode_conflict_samples)),
            "avg_conflict_percent": float(100.0 * np.mean(episode_conflict_samples)),
            "epsilon": float(eps),
            "policy_loss_mean": float(np.mean(policy_losses)) if len(policy_losses) > 0 else np.nan,
            "critic_loss_mean": float(np.mean(critic_losses)) if len(critic_losses) > 0 else np.nan,
            "dr_loss_mean": float(np.mean(dr_losses)) if len(dr_losses) > 0 else np.nan,
            "dr_mixed_reward_mean": float(np.mean(dr_mixed_reward_vals)) if len(dr_mixed_reward_vals) > 0 else np.nan,
            "u_solver_mean": float(np.mean(dr_u_solver_vals)) if len(dr_u_solver_vals) > 0 else np.nan,
            "u_neutral_mean": float(np.mean(dr_u_neutral_vals)) if len(dr_u_neutral_vals) > 0 else np.nan,
            "u_causer_mean": float(np.mean(dr_u_causer_vals)) if len(dr_u_causer_vals) > 0 else np.nan,
            "v_c_mean": float(np.mean(dr_v_c_vals)) if len(dr_v_c_vals) > 0 else np.nan,
            "v_p_mean": float(np.mean(dr_v_p_vals)) if len(dr_v_p_vals) > 0 else np.nan,
            "v_mean_mean": float(np.mean(dr_v_mean_vals)) if len(dr_v_mean_vals) > 0 else np.nan,
            "my_mean_mean": float(np.mean(dr_my_mean_vals)) if len(dr_my_mean_vals) > 0 else np.nan,
            "arm_wait3_count": np.nan,
            "arm_backward3_count": np.nan,
            "arm_randomwalk3_count": np.nan,
            "arm_wait2_forward1_count": np.nan,
            "arm_move_clear_count": np.nan,
            "arm_handshake_count": np.nan,
            "arm_reserve_parity_count": np.nan,
            "arm_reserve_parity_escape_count": np.nan,
            "arm_priority_swap_n3_count": np.nan,
            "arm_pass_food_n3_count": np.nan,
            "arm_freeze_tag_count": np.nan,
            "arm_wait3_prob": np.nan,
            "arm_backward3_prob": np.nan,
            "arm_randomwalk3_prob": np.nan,
            "arm_wait2_forward1_prob": np.nan,
            "arm_move_clear_prob": np.nan,
            "arm_handshake_prob": np.nan,
            "arm_reserve_parity_prob": np.nan,
            "arm_reserve_parity_escape_prob": np.nan,
            "arm_priority_swap_n3_prob": np.nan,
            "arm_pass_food_n3_prob": np.nan,
            "arm_freeze_tag_prob": np.nan,
        }
        episode_metrics.append(metric_row)
        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_env_rewards[-50:])
            print(f"Episode {episode + 1}/{n_episodes} | Avg env reward (last 50): {avg:.2f} | Epsilon: {eps:.3f}")

    for env_i in envs:
        env_i.close()
    print(
        "Training complete. "
        f"Final avg env reward per env: {np.mean(episode_env_rewards[-50:]):.2f} | "
        f"raw total across envs: {np.mean(episode_env_rewards_raw[-50:]):.2f}"
    )

    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", newline="", encoding="utf-8") as csv_file:
        if len(episode_metrics) == 0:
            raise ValueError("episode_metrics is unexpectedly empty")
        writer = csv.DictWriter(csv_file, fieldnames=list(episode_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(episode_metrics)
    print(f"Saved training metrics to {metrics_path}")

    artifacts_dir = metrics_path.parent
    for agent_id in env0.possible_agents:
        torch.save(policies[agent_id].q_net.state_dict(), artifacts_dir / f"dqn_weights_{agent_id}.pt")
        torch.save(critics[agent_id].net.state_dict(), artifacts_dir / f"pc_critic_{agent_id}.pt")
    print(f"Saved {n_agents} policies and critics under {artifacts_dir}")

    if args.reward_mode == "scenario_mixture":
        if len(dr_learners) == 0:
            raise ValueError("scenario_mixture requires non-empty dr_learners at save time")
        per_agent_meta = {}
        for agent_id in env0.possible_agents:
            learner = dr_learners[agent_id]
            torch.save(learner.net.state_dict(), artifacts_dir / f"dr_mixture_{agent_id}.pt")
            if learner.running_own_p_count <= 0:
                raise ValueError(f"DR learner {agent_id} has zero running_own_p_count; cannot save meta")
            per_agent_meta[agent_id] = {
                "running_own_p_sum": float(learner.running_own_p_sum),
                "running_own_p_count": int(learner.running_own_p_count),
            }
        meta = {"full_duration": float(FULL_DURATION_NORM), "agents": per_agent_meta}
        meta_path = artifacts_dir / "dr_mixture_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved DR mixture nets and {meta_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

