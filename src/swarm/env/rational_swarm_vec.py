"""
Batched NumPy implementation of RationalSwarmForagingEnv.

`num_envs` replicas share one step/reset call; dynamics follow ``rational_swarm.py``
(rules aligned; each row uses its own ``numpy.random.RandomState`` so refills do not
cross-contaminate RNG across rows, unlike a single shared ``np.random`` stream).

This is **single-threaded** array batching (one interpreter, no worker processes).
Larger ``num_envs`` increases work per ``step()``; it does not parallelize simulation
across CPU cores. Throughput gains come from amortizing Python/NumPy overhead and
from feeding wider batches to downstream code, not from multithreading.
"""

from __future__ import annotations

import numpy as np

from swarm.env.rational_swarm import normalized_planar_position_xy


class RationalSwarmForagingVecEnv:
    """
    Vectorized foraging env: state shape (num_envs, n_agents, ...).

    API:
      - ``possible_agents``: list[str]
      - ``reset(seeds)`` -> observations dict, infos dict
      - ``step(actions)`` -> observations, rewards, terminations, truncations, infos
    Each ``actions[agent_id]`` is ``(num_envs,)`` int32 in ``[0, 4]``.
    Observations are ``(num_envs, obs_dim)`` float32 per agent (tile channels in ``[0, 5]``, then
    normalized planar position in ``[0, 2]``, then carrying in ``{0, 1}``).
    """

    def __init__(
        self,
        num_envs: int,
        n_agents: int = 3,
        grid_size: int = 10,
        num_food: int = 5,
        local_grid_size: int = 5,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        self.num_envs = int(num_envs)
        self.n_agents = int(n_agents)
        self.grid_size = int(grid_size)
        self.num_food = int(num_food)
        self.local_grid_size = int(local_grid_size)
        self.G = self.grid_size
        self.A = self.n_agents
        self.N = self.num_envs
        self.K = self.num_food
        self.L = self.local_grid_size
        self.hw = self.L // 2
        self.obs_dim = self.L * self.L + 2 + 1

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.base_xy = np.array([self.G // 2, self.G // 2], dtype=np.int32)

        self._pos = np.zeros((self.N, self.A, 2), dtype=np.int32)
        self._holding = np.zeros((self.N, self.A), dtype=np.bool_)
        self._avoid_rem = np.zeros((self.N, self.A), dtype=np.int32)
        self._p_time = np.zeros((self.N, self.A), dtype=np.int32)
        self._c_time = np.zeros((self.N, self.A), dtype=np.int32)
        self._food_xy = np.zeros((self.N, self.K, 2), dtype=np.int32)
        self._food_alive = np.zeros((self.N, self.K), dtype=np.bool_)
        self._obs_mode_prog = np.ones((self.N, self.A), dtype=np.bool_)
        self._rng: list[np.random.RandomState] = [np.random.RandomState(0) for _ in range(self.N)]

    def _reset_row(self, n: int) -> None:
        """Sample foods and agents for row ``n`` using ``self._rng[n]`` (legacy MT19937 ``randint`` draws)."""
        rs = self._rng[n]
        bx, by = int(self.base_xy[0]), int(self.base_xy[1])
        base_list = [bx, by]
        foods: list[list[int]] = []
        while len(foods) < self.K:
            pos = [int(rs.randint(0, self.G)), int(rs.randint(0, self.G))]
            if pos != base_list and pos not in foods:
                foods.append(pos)
        for k in range(self.K):
            self._food_xy[n, k, 0] = foods[k][0]
            self._food_xy[n, k, 1] = foods[k][1]
            self._food_alive[n, k] = True
        used = {tuple(base_list)}
        for f in foods:
            used.add(tuple(f))
        for a in range(self.A):
            while True:
                pos = [int(rs.randint(0, self.G)), int(rs.randint(0, self.G))]
                t = tuple(pos)
                if t not in used:
                    used.add(t)
                    self._pos[n, a, 0] = pos[0]
                    self._pos[n, a, 1] = pos[1]
                    break

    def reset(self, seeds: np.ndarray):
        """
        Reset all replicas. ``seeds`` shape ``(num_envs,)`` — one integer seed per row.
        """
        seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
        if seeds.shape[0] != self.N:
            raise ValueError(f"seeds must have shape ({self.N},), got {seeds.shape}")

        self._holding[:] = False
        self._avoid_rem[:] = 0
        self._p_time[:] = 0
        self._c_time[:] = 0

        for n in range(self.N):
            self._rng[n] = np.random.RandomState(int(seeds[n]))
            self._food_alive[n, :] = False
            self._reset_row(n)

        self._obs_mode_prog[:] = self._avoid_rem == 0
        obs = self._build_observations()
        infos = {agent: {} for agent in self.possible_agents}
        return obs, infos

    def _refill_foods_row(self, n: int) -> None:
        """Refill dead food slots to ``K`` alive (same draws as ``RationalSwarmForagingEnv.step``)."""
        bx, by = int(self.base_xy[0]), int(self.base_xy[1])
        base_list = [bx, by]
        pos_flat = self._pos[n]
        rs = self._rng[n]
        while int(self._food_alive[n].sum()) < self.K:
            new_f = [int(rs.randint(0, self.G)), int(rs.randint(0, self.G))]
            if new_f == base_list:
                continue
            ok = True
            for k in range(self.K):
                if self._food_alive[n, k] and int(self._food_xy[n, k, 0]) == new_f[0] and int(self._food_xy[n, k, 1]) == new_f[1]:
                    ok = False
                    break
            if not ok:
                continue
            for a in range(self.A):
                if int(pos_flat[a, 0]) == new_f[0] and int(pos_flat[a, 1]) == new_f[1]:
                    ok = False
                    break
            if not ok:
                continue
            for k in range(self.K):
                if not self._food_alive[n, k]:
                    self._food_xy[n, k, 0] = new_f[0]
                    self._food_xy[n, k, 1] = new_f[1]
                    self._food_alive[n, k] = True
                    break

    def _build_observations(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for a in range(self.A):
            out[self.possible_agents[a]] = self._obs_for_agent(a)
        return out

    def _obs_for_agent(self, a: int) -> np.ndarray:
        """Uses ``self._obs_mode_prog`` (N, A): True iff agent is in program mode for tile coding."""
        mode_prog = self._obs_mode_prog
        N = self.N
        ax = self._pos[:, a, 0]
        ay = self._pos[:, a, 1]
        base_x, base_y = int(self.base_xy[0]), int(self.base_xy[1])
        L = self.L
        jj = np.arange(L, dtype=np.int32)
        ii = np.arange(L, dtype=np.int32)
        wx1 = ax[:, None, None] - self.hw + jj[None, None, :]
        wy1 = ay[:, None, None] - self.hw + ii[None, :, None]
        wx, wy = np.broadcast_arrays(wx1, wy1)
        outb = (wx < 0) | (wx >= self.G) | (wy < 0) | (wy >= self.G)
        on_base = (wx == base_x) & (wy == base_y)

        fxk = self._food_xy[:, :, 0][:, :, None, None]
        fyk = self._food_xy[:, :, 1][:, :, None, None]
        alive_k = self._food_alive[:, :, None, None]
        food_match = alive_k & (wx[:, None, :, :] == fxk) & (wy[:, None, :, :] == fyk)
        food_here = food_match.any(axis=1)

        other_val = np.zeros((N, L, L), dtype=np.int32)
        for a2 in range(self.A):
            if a2 == a:
                continue
            px = self._pos[:, a2, 0][:, None, None]
            py = self._pos[:, a2, 1][:, None, None]
            same = (px == wx) & (py == wy)
            val = np.where(mode_prog[:, a2, None, None], 3, 4)
            replace = same & ((other_val == 0) | mode_prog[:, a2, None, None])
            other_val = np.where(replace, val, other_val)

        cell = np.zeros((N, L, L), dtype=np.int32)
        cell = np.where(outb, 5, 0)
        cell = np.where(~outb & on_base, 1, cell)
        cell = np.where(~outb & ~on_base & food_here, 2, cell)
        cell = np.where(~outb & ~on_base & ~food_here & (other_val > 0), other_val, cell)
        tiles = cell.reshape(N, L * L)

        nx, ny = normalized_planar_position_xy(self._pos[:, a, 0], self._pos[:, a, 1], self.G)
        hold_f = self._holding[:, a].astype(np.float32)
        tiles_f = tiles.astype(np.float32)
        return np.concatenate([tiles_f, nx[:, None], ny[:, None], hold_f[:, None]], axis=1)

    def _n_local_from_obs(self, obs: np.ndarray) -> np.ndarray:
        """obs (N, obs_dim) -> (N,) counts."""
        n_tiles = self.L * self.L
        grid = obs[:, :n_tiles].reshape(-1, self.L, self.L)
        c = self.L // 2
        adj = np.stack(
            [
                grid[:, c - 1, c],
                grid[:, c + 1, c],
                grid[:, c, c - 1],
                grid[:, c, c + 1],
            ],
            axis=1,
        )
        other_adj = ((adj == 3) | (adj == 4)).sum(axis=1)
        return 1 + other_adj.astype(np.int32)

    def step(self, actions: dict[str, np.ndarray]):
        act = np.stack([actions[name] for name in self.possible_agents], axis=1)
        act = np.asarray(act, dtype=np.int32)
        if act.shape != (self.N, self.A):
            raise ValueError(f"actions stacked shape must be ({self.N}, {self.A}), got {act.shape}")

        pos = self._pos
        avoid_rem = self._avoid_rem.copy()
        rewards = np.zeros((self.N, self.A), dtype=np.float64)

        lock = avoid_rem > 0
        avoid_rem = np.where(lock, avoid_rem - 1, avoid_rem)

        noop_branch = (~lock) & (act == 4)
        program_move = (~lock) & (~noop_branch)

        next_pos = pos.copy()
        mode_prog = np.zeros((self.N, self.A), dtype=np.bool_)

        next_pos = np.where(lock[..., None], pos, next_pos)
        mode_prog = np.where(lock, False, mode_prog)
        self._c_time = np.where(lock, self._c_time + 1, self._c_time)

        avoid_rem = np.where(noop_branch, 2, avoid_rem)
        next_pos = np.where(noop_branch[..., None], pos, next_pos)
        mode_prog = np.where(noop_branch, False, mode_prog)
        self._c_time = np.where(noop_branch, self._c_time + 1, self._c_time)

        nx = pos[:, :, 0]
        ny = pos[:, :, 1]
        d0 = (act == 0) & program_move & (nx > 0)
        d1 = (act == 1) & program_move & (nx < self.G - 1)
        d2 = (act == 2) & program_move & (ny > 0)
        d3 = (act == 3) & program_move & (ny < self.G - 1)
        new_x = nx - d0.astype(np.int32) + d1.astype(np.int32)
        new_y = ny - d2.astype(np.int32) + d3.astype(np.int32)
        proposed = np.stack([new_x, new_y], axis=-1)
        next_pos = np.where(program_move[..., None], proposed, next_pos)
        mode_prog = np.where(program_move, True, mode_prog)
        self._p_time = np.where(program_move, self._p_time + 1, self._p_time)

        cell_id = next_pos[..., 0] * self.G + next_pos[..., 1]
        same = cell_id[:, :, None] == cell_id[:, None, :]
        cnt = same.sum(axis=2)
        multi = cnt > 1

        conflict_program = multi & mode_prog
        self._p_time = np.where(conflict_program, self._p_time - 1, self._p_time)
        self._c_time = np.where(conflict_program, self._c_time + 1, self._c_time)
        avoid_rem = np.where(conflict_program, 2, avoid_rem)
        mode_prog = np.where(conflict_program, False, mode_prog)

        pos_new = np.where((~multi)[..., None], next_pos, pos)
        self._pos = pos_new
        self._avoid_rem = avoid_rem
        self._obs_mode_prog = mode_prog.copy()

        bx, by = int(self.base_xy[0]), int(self.base_xy[1])
        for n in range(self.N):
            for a in range(self.A):
                if self._holding[n, a] and int(self._pos[n, a, 0]) == bx and int(self._pos[n, a, 1]) == by:
                    self._holding[n, a] = False
                    rewards[n, a] += 1.0
                    self._refill_foods_row(n)

        for n in range(self.N):
            for a in range(self.A):
                if self._holding[n, a]:
                    continue
                px, py = int(self._pos[n, a, 0]), int(self._pos[n, a, 1])
                for k in range(self.K):
                    if not self._food_alive[n, k]:
                        continue
                    fx, fy = int(self._food_xy[n, k, 0]), int(self._food_xy[n, k, 1])
                    if px == fx and py == fy:
                        self._food_alive[n, k] = False
                        self._holding[n, a] = True
                        break

        env_reward = rewards.sum(axis=1)
        obs = self._build_observations()

        terminations = {agent: np.zeros(self.N, dtype=np.bool_) for agent in self.possible_agents}
        truncations = {agent: np.zeros(self.N, dtype=np.bool_) for agent in self.possible_agents}
        rew_dict = {self.possible_agents[a]: rewards[:, a].copy() for a in range(self.A)}

        infos: dict[str, dict] = {}
        for a, agent in enumerate(self.possible_agents):
            o = obs[agent]
            mp = self._obs_mode_prog[:, a]
            p_t = mp.astype(np.float32)
            c_t = (~mp).astype(np.float32)
            infos[agent] = {
                "env_reward": env_reward.astype(np.float64),
                "p_t": p_t,
                "c_t": c_t,
                "n_local": self._n_local_from_obs(o).astype(np.float32),
            }
        return obs, rew_dict, terminations, truncations, infos

    def close(self) -> None:
        pass
