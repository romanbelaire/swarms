import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
import pygame

from swarm.config import ENV_LAYOUT_NAMES


def normalized_planar_position_xy(ax: np.ndarray, ay: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Map integer cell indices [0, grid_size-1] to [0, 2] per axis: subtract grid midpoint (Cartesian center),
    divide by the half-span to [-1, 1], then shift/scale to [0, 2]. Invariant to absolute grid extent
    when combined with the same grid_size at train and eval time (relative geometry matches relative encoding).
    """
    half_span = (grid_size - 1) * 0.5
    if half_span == 0:
        raise ValueError("normalized_planar_position_xy requires grid_size >= 2")
    ax64 = np.asarray(ax, dtype=np.float64).reshape(-1)
    ay64 = np.asarray(ay, dtype=np.float64).reshape(-1)
    mx = ax64 - half_span
    my = ay64 - half_span
    nx = 1.0 + mx / half_span
    ny = 1.0 + my / half_span
    return nx.astype(np.float32), ny.astype(np.float32)


FREEZE_TAG_TURNS = 3


def base_positions_for_layout(grid_size: int, env_layout: str) -> tuple[tuple[int, int], ...]:
    if env_layout not in ENV_LAYOUT_NAMES:
        raise ValueError(f"env_layout must be one of {ENV_LAYOUT_NAMES}, got {env_layout!r}")
    if env_layout == "center_base":
        return ((grid_size // 2, grid_size // 2),)
    mid = grid_size // 2
    c_lo = (mid - 1) // 2
    c_hi = (mid + grid_size - 1) // 2
    return ((c_lo, c_lo), (c_hi, c_hi))


class RationalSwarmForagingEnv(ParallelEnv):
    """
    A 2D Grid Foraging Environment using the Rational Swarms Difference Reward Model (Kaminka et al. 2025).

    **Action space** (per agent): Discrete(5) — same for every agent.

    - 0 / 1: move along the first grid axis (decrease / increase agent position index 0).
    - 2 / 3: move along the second grid axis (decrease / increase agent position index 1).
    - 4: no-op.

    Moves that would leave the grid are ignored (position unchanged). If two agents try to program
    into the same cell, both are forced into avoidance with a short lockout (see step).

    **Observation space** (per agent): Box with ``dtype=float32``, shape ``(obs_dim,)`` where
    ``obs_dim = local_grid_size ** 2 + 3`` (tiles + planar position + carry).

    immunity / freeze durations are exposed in ``infos``, not appended to observations, so frozen
    task experts trained at the legacy ``OBS_DIM`` still load without shape mismatch.

    **Freeze / immunity:** ``agent_freeze_remaining`` freezes movement and counts each frozen step as
    conflict (``C_time`` / ``c_t``). ``agent_immune_priority`` is
    ``0`` normally; when immunity is granted it becomes a positive ticket (tie-break elsewhere).
    Immune agents run **as a perceptual ghost for one food haul**: observations mask all other
    agents as empty tiles (base/food/obstacle framing unchanged); they skip avoidance lockout,
    positional ``position_claims`` mediation, and apply their move intent unconditionally (overlap
    allowed). Training conflict flags keyed off ``n_local`` thus stay off while immune (no partners in
    the local grid). Immunity clears on successful deposit at the base.

    Non-immune agents resolve shared cells as usual; immune agents never appear in ``position_claims``.

    **``suppress_agent_collision``:** when true, positional conflict resolution between non-immune
    agents is skipped: every agent occupies its ``next_positions`` intent, overlaps allowed (except
    food/base logic unchanged).

    **Rewards:** rewards[agent] += 1 when that agent deposits food at a base. Training code uses infos[agent] ("env_reward", "p_t", "c_t", "n_local") for shaping and logging.

    **``env_layout``:** ``center_base`` places one base at the grid center. ``dual_quadrant_base`` places bases at
    the centers of the low–low and high–high quadrants (split at ``grid_size // 2``), e.g. (2, 2) and (7, 7) when
    ``grid_size=10``, so foraging does not pull agents to the geometric center.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "rational_swarms_foraging_v0",
    }

    def __init__(
        self,
        n_agents=3,
        grid_size=10,
        num_food=5,
        alpha=1.0,
        beta=1.0,
        local_grid_size=5,
        render_mode=None,
        suppress_agent_collision: bool = False,
        env_layout: str = "center_base",
    ):
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.num_food = num_food
        self.alpha = alpha
        self.beta = beta
        self.local_grid_size = local_grid_size
        self.render_mode = render_mode
        self.suppress_agent_collision = suppress_agent_collision
        self.env_layout = env_layout
        self.base_positions = base_positions_for_layout(self.grid_size, env_layout)
        self._base_position_set = {tuple(p) for p in self.base_positions}
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.local_grid_size <= 0:
            raise ValueError("local_grid_size must be positive")
        if self.num_food < 0:
            raise ValueError("num_food must be non-negative")
        max_non_base_cells = self.grid_size * self.grid_size - len(self.base_positions)
        if self.n_agents > max_non_base_cells:
            raise ValueError(
                f"n_agents={self.n_agents} exceeds capacity for grid_size={self.grid_size}; "
                f"max supported is {max_non_base_cells}"
            )
        if self.num_food > max_non_base_cells:
            raise ValueError(
                f"num_food={self.num_food} exceeds capacity for grid_size={self.grid_size}; "
                f"max supported is {max_non_base_cells}"
            )

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(self.n_agents))))
        self._action_spaces = {agent: Discrete(5) for agent in self.possible_agents}

        obs_dim = self.local_grid_size ** 2 + 2 + 1
        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.zeros(obs_dim, dtype=np.float32)
        high[: self.local_grid_size**2] = 5.0
        high[self.local_grid_size**2 : self.local_grid_size**2 + 2] = 2.0
        high[-1] = 1.0
        self._observation_spaces = {agent: Box(low=low, high=high, shape=(obs_dim,), dtype=np.float32) for agent in self.possible_agents}

        self.agents = []
        self.agent_positions = {}
        self.agent_holding_food = {}
        self.food_positions = []

        self.agent_modes = {}
        self.agent_avoidance_remaining = {}
        self.P_time = {}
        self.C_time = {}
        self.agent_freeze_remaining = {}
        self.agent_immune_priority = {}
        self._immune_ticket_seq = 0

        self.window = None
        self.clock = None
        self.cell_size = 50
        self.window_size = self.grid_size * self.cell_size

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]

        self.food_positions = []
        while len(self.food_positions) < self.num_food:
            pos = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
            if tuple(pos) not in self._base_position_set and pos not in self.food_positions:
                self.food_positions.append(pos)

        used = set(self._base_position_set)
        self.agent_positions = {}
        for agent in self.agents:
            while True:
                pos = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
                if tuple(pos) not in used:
                    used.add(tuple(pos))
                    self.agent_positions[agent] = pos
                    break

        self.agent_modes = {agent: "program" for agent in self.agents}
        self.agent_avoidance_remaining = {agent: 0 for agent in self.agents}
        self.P_time = {agent: 0 for agent in self.agents}
        self.C_time = {agent: 0 for agent in self.agents}
        self.agent_holding_food = {agent: False for agent in self.agents}
        self.agent_freeze_remaining = {agent: 0 for agent in self.agents}
        self.agent_immune_priority = {agent: 0 for agent in self.agents}
        self._immune_ticket_seq = 0

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_obs(self, agent):
        ax, ay = self.agent_positions[agent]
        hw = self.local_grid_size // 2
        food_set = {tuple(f) for f in self.food_positions}
        other_agent_at = {}
        if self.agent_immune_priority[agent] <= 0:
            for a in self.agents:
                if a == agent:
                    continue
                pos = tuple(self.agent_positions[a])
                val = 3 if self.agent_modes[a] == "program" else 4
                if pos not in other_agent_at:
                    other_agent_at[pos] = val
                elif val == 3:
                    other_agent_at[pos] = 3

        grid_flat = []
        for i in range(self.local_grid_size):
            for j in range(self.local_grid_size):
                wx, wy = ax - hw + j, ay - hw + i
                if wx < 0 or wx >= self.grid_size or wy < 0 or wy >= self.grid_size:
                    grid_flat.append(5)
                elif (wx, wy) in self._base_position_set:
                    grid_flat.append(1)
                elif (wx, wy) in food_set:
                    grid_flat.append(2)
                elif (wx, wy) in other_agent_at:
                    grid_flat.append(other_agent_at[(wx, wy)])
                else:
                    grid_flat.append(0)

        holding = 1.0 if self.agent_holding_food[agent] else 0.0
        nx, ny = normalized_planar_position_xy(np.array([ax]), np.array([ay]), self.grid_size)
        tiles = np.asarray(grid_flat, dtype=np.float32)
        obs = np.concatenate([tiles, nx, ny, np.array([holding], dtype=np.float32)])
        return obs

    def _orthogonal_neighbor_agent_ids(self, agent: str) -> list[str]:
        ax, ay = self.agent_positions[agent]
        out = []
        for other_id in self.agents:
            if other_id == agent:
                continue
            ox, oy = self.agent_positions[other_id]
            if ox == ax - 1 and oy == ay:
                out.append(other_id)
            elif ox == ax + 1 and oy == ay:
                out.append(other_id)
            elif ox == ax and oy == ay - 1:
                out.append(other_id)
            elif ox == ax and oy == ay + 1:
                out.append(other_id)
        return out

    def apply_freeze_tag_pair(self, agent_a: str, agent_b: str):
        self.agent_freeze_remaining[agent_a] = FREEZE_TAG_TURNS
        self.agent_freeze_remaining[agent_b] = FREEZE_TAG_TURNS

    def grant_immunity_unlock_bulk(self, agent_ids: tuple[str, ...]):
        for agent in agent_ids:
            self.agent_freeze_remaining[agent] = 0
            self.agent_avoidance_remaining[agent] = 0
            self._immune_ticket_seq += 1
            self.agent_immune_priority[agent] = self._immune_ticket_seq

    def _frozen_nonimmune_blocks_cell(self, pos_tuple: tuple[int, int], mover_id: str) -> bool:
        for a in self.agents:
            if a == mover_id:
                continue
            if self.agent_freeze_remaining[a] <= 0:
                continue
            if self.agent_immune_priority[a] > 0:
                continue
            if tuple(self.agent_positions[a]) == pos_tuple:
                return True
        return False

    def _n_local_neighborhood(self, obs: np.ndarray) -> int:
        # Conflict semantics: only orthogonally adjacent agents count as local neighbors.
        n_tiles = self.local_grid_size ** 2
        grid_flat = obs[:n_tiles].reshape(self.local_grid_size, self.local_grid_size)
        c = self.local_grid_size // 2
        adjacent_cells = [grid_flat[c - 1, c], grid_flat[c + 1, c], grid_flat[c, c - 1], grid_flat[c, c + 1]]
        other_adjacent = sum(int(cell == 3 or cell == 4) for cell in adjacent_cells)
        return 1 + other_adjacent

    def step(self, actions):
        rewards = {agent: 0 for agent in self.agents}

        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        prev_positions = {agent: list(self.agent_positions[agent]) for agent in self.agents}

        for agent in self.agents:
            if self.agent_freeze_remaining[agent] <= 0:
                continue
            frozen_neighbors = 0
            for nid in self._orthogonal_neighbor_agent_ids(agent):
                if self.agent_freeze_remaining[nid] > 0:
                    frozen_neighbors += 1
            if frozen_neighbors >= 2:
                self.agent_freeze_remaining[agent] = 0
                self.agent_avoidance_remaining[agent] = 0
                self._immune_ticket_seq += 1
                self.agent_immune_priority[agent] = self._immune_ticket_seq

        next_positions = {}
        for agent in self.agents:
            action = actions.get(agent, 4)
            current_pos = self.agent_positions[agent]

            if self.agent_freeze_remaining[agent] > 0:
                self.agent_modes[agent] = "avoidance"
                self.C_time[agent] += 1
                next_positions[agent] = list(current_pos)
                continue

            if self.agent_immune_priority[agent] > 0:
                self.agent_modes[agent] = "program"
                self.P_time[agent] += 1
                if action == 4:
                    next_positions[agent] = list(current_pos)
                else:
                    new_pos = list(current_pos)
                    if action == 0 and new_pos[0] > 0:
                        new_pos[0] -= 1
                    elif action == 1 and new_pos[0] < self.grid_size - 1:
                        new_pos[0] += 1
                    elif action == 2 and new_pos[1] > 0:
                        new_pos[1] -= 1
                    elif action == 3 and new_pos[1] < self.grid_size - 1:
                        new_pos[1] += 1
                    next_positions[agent] = new_pos
                continue

            if self.agent_avoidance_remaining[agent] > 0:
                self.agent_avoidance_remaining[agent] -= 1
                self.agent_modes[agent] = "avoidance"
                self.C_time[agent] += 1
                next_positions[agent] = current_pos
            elif action == 4:
                self.agent_avoidance_remaining[agent] = 2
                self.agent_modes[agent] = "avoidance"
                self.C_time[agent] += 1
                next_positions[agent] = current_pos
            else:
                self.agent_modes[agent] = "program"
                self.P_time[agent] += 1
                new_pos = list(current_pos)
                if action == 0 and new_pos[0] > 0:
                    new_pos[0] -= 1
                elif action == 1 and new_pos[0] < self.grid_size - 1:
                    new_pos[0] += 1
                elif action == 2 and new_pos[1] > 0:
                    new_pos[1] -= 1
                elif action == 3 and new_pos[1] < self.grid_size - 1:
                    new_pos[1] += 1
                next_positions[agent] = new_pos

        self._collision_conflict_event = {agent: False for agent in self.agents}

        if self.suppress_agent_collision:
            for agent in self.agents:
                self.agent_positions[agent] = list(next_positions[agent])
        else:
            for agent in self.agents:
                if self.agent_immune_priority[agent] > 0:
                    self.agent_positions[agent] = list(next_positions[agent])

            for agent in self.agents:
                if self.agent_freeze_remaining[agent] > 0:
                    self.agent_positions[agent] = list(next_positions[agent])

            position_claims: dict[tuple[int, int], list[str]] = {}
            for agent in self.agents:
                if self.agent_immune_priority[agent] > 0:
                    continue
                if self.agent_freeze_remaining[agent] > 0:
                    continue
                pos_tuple = tuple(next_positions[agent])
                if pos_tuple not in position_claims:
                    position_claims[pos_tuple] = []
                position_claims[pos_tuple].append(agent)

            for pos_tuple, claiming_agents in position_claims.items():
                if len(claiming_agents) > 1:
                    for agent in claiming_agents:
                        if self.agent_modes[agent] == "program":
                            self.P_time[agent] -= 1
                            self.C_time[agent] += 1
                            self.agent_modes[agent] = "avoidance"
                            self.agent_avoidance_remaining[agent] = 2
                        self._collision_conflict_event[agent] = True
                        self.agent_positions[agent] = prev_positions[agent]
                    continue

                lone = claiming_agents[0]
                target = tuple(next_positions[lone])
                prev_tuple = tuple(prev_positions[lone])
                if (
                    target != prev_tuple
                    and self._frozen_nonimmune_blocks_cell(target, lone)
                ):
                    if self.agent_modes[lone] == "program":
                        self.P_time[lone] -= 1
                        self.C_time[lone] += 1
                        self.agent_modes[lone] = "avoidance"
                        self.agent_avoidance_remaining[lone] = 2
                    self._collision_conflict_event[lone] = True
                    self.agent_positions[lone] = prev_positions[lone]
                else:
                    self.agent_positions[lone] = next_positions[lone]

        for agent in sorted(self.agents):
            if self.agent_holding_food[agent] and tuple(self.agent_positions[agent]) in self._base_position_set:
                self.agent_holding_food[agent] = False
                self.agent_immune_priority[agent] = 0
                rewards[agent] += 1.0
                while len(self.food_positions) < self.num_food:
                    new_f = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
                    if tuple(new_f) not in self._base_position_set and new_f not in self.food_positions:
                        self.food_positions.append(new_f)

        for agent in sorted(self.agents):
            if not self.agent_holding_food[agent] and tuple(self.agent_positions[agent]) in [tuple(f) for f in self.food_positions]:
                self.food_positions = [f for f in self.food_positions if tuple(f) != tuple(self.agent_positions[agent])]
                self.agent_holding_food[agent] = True

        for agent in self.agents:
            if self.agent_freeze_remaining[agent] > 0:
                self.agent_freeze_remaining[agent] -= 1

        env_reward = sum(rewards.values())
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {}
        for agent in self.agents:
            o = observations[agent]
            p_t = int(self.agent_modes[agent] == "program")
            c_t = int(self.agent_modes[agent] == "avoidance")
            infos[agent] = {
                "env_reward": env_reward,
                "p_t": p_t,
                "c_t": c_t,
                "n_local": self._n_local_neighborhood(o),
                "new_conflict_event": float(self._collision_conflict_event[agent]),
                "freeze_remaining": float(self.agent_freeze_remaining[agent]),
                "immune_priority": float(self.agent_immune_priority[agent]),
                "is_immune": float(int(self.agent_immune_priority[agent] > 0)),
            }

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Rational Swarm Foraging Env")
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(canvas, (200, 200, 200), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(canvas, (200, 200, 200), (0, y), (self.window_size, y))

        for base_pos in self.base_positions:
            base_rect = pygame.Rect(
                base_pos[1] * self.cell_size,
                base_pos[0] * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            pygame.draw.rect(canvas, (0, 0, 255), base_rect)

        for f in self.food_positions:
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                (f[1] * self.cell_size + self.cell_size // 2, f[0] * self.cell_size + self.cell_size // 2),
                self.cell_size // 4,
            )

        for idx, agent in enumerate(self.agents):
            pos = self.agent_positions[agent]
            mode = self.agent_modes[agent]
            holding = self.agent_holding_food[agent]
            immune = self.agent_immune_priority[agent] > 0
            frozen = self.agent_freeze_remaining[agent] > 0
            if immune:
                color = (0, 200, 255)
            elif frozen:
                color = (160, 160, 220)
            else:
                color = (255, 0, 0) if mode == "avoidance" else (0, 0, 0)

            offset_x = (idx % 3 - 1) * 5
            offset_y = (idx // 3) * 5
            cx = pos[1] * self.cell_size + self.cell_size // 2 + offset_x
            cy = pos[0] * self.cell_size + self.cell_size // 2 + offset_y

            pygame.draw.circle(canvas, color, (cx, cy), self.cell_size // 3)
            if holding:
                pygame.draw.circle(canvas, (255, 200, 0), (cx, cy), self.cell_size // 6)
            font = pygame.font.SysFont(None, 24)
            img = font.render(str(idx), True, (255, 255, 255))
            canvas.blit(img, (cx - 5, cy - 8))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(60)
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

