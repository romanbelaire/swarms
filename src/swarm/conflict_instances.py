"""Conflict-instance P/C accounting with length-normalized aggregates."""

from dataclasses import dataclass

import numpy as np

MAX_CONFLICT_STEPS_TYPE_NAMES = ("time", "distance")


def step_team_utility_mean(infos: dict[str, dict], agent_ids: list[str]) -> float:
    """Per-step observed team util: mean over agents of (p_t - c_t), in [-1, 1]."""
    return float(
        sum(float(infos[agent_id]["p_t"]) - float(infos[agent_id]["c_t"]) for agent_id in agent_ids)
        / len(agent_ids)
    )


@dataclass(frozen=True)
class ClosedConflictInstance:
    p_norm: float
    c_norm: float
    team_util_norm: float
    length: int


class ConflictInstanceTracker:
    """
    Tracks one conflict episode per agent: avoidance while in a local conflict,
    then program time (including immune haul) until the next conflict entry.

    - **Start:** ``n_local > 1`` opens an instance in the avoidance phase (agent
      should be in avoidance / escaping).
    - **Escape:** ``n_local <= 1`` while the instance is active switches to the
      program-haul phase; ``p_t`` steps accumulate (immune haul counts).
    - **End:** ``n_local > 1`` again closes the instance (``D`` uses normalized
      totals), then a new avoidance instance opens on that step.
    - **Escape timeout:** if still in conflict after ``max_conflict_steps`` in the
      avoidance phase, force-close and reopen a fresh avoidance instance (current
      step only in the new window).
    """

    def __init__(self, max_conflict_steps: int, max_conflict_steps_type: str):
        if max_conflict_steps < 0:
            raise ValueError("max_conflict_steps must be non-negative")
        if max_conflict_steps_type not in MAX_CONFLICT_STEPS_TYPE_NAMES:
            raise ValueError(
                f"max_conflict_steps_type must be one of {MAX_CONFLICT_STEPS_TYPE_NAMES}; "
                f"got {max_conflict_steps_type}"
            )
        self._max_conflict_steps = int(max_conflict_steps)
        self._max_conflict_steps_type = max_conflict_steps_type
        self._active = False
        self._phase = ""
        self._p = 0
        self._c = 0
        self._team_util = 0.0
        self._unmoved_steps = 0
        self._closed: list[ClosedConflictInstance] = []

    def step(
        self,
        *,
        p_t: float,
        c_t: float,
        team_util: float,
        n_local: float,
        new_conflict_event: float,
        agent_moved: bool,
    ) -> ClosedConflictInstance | None:
        in_conflict = n_local > 1.0
        closed: ClosedConflictInstance | None = None

        if new_conflict_event > 0.5 and self._active:
            closed = self._close()

        if self._active and self._phase == "program_haul" and in_conflict:
            closed = self._close()

        if in_conflict:
            if self._active and self._phase == "avoidance" and self._should_split_avoidance(agent_moved):
                closed = self._split_avoidance(p_t, c_t, team_util, agent_moved)
            elif not self._active:
                self._open_avoidance(p_t, c_t, team_util, agent_moved)
            elif self._phase == "avoidance":
                self._accumulate(p_t, c_t, team_util, agent_moved)
            else:
                self._open_avoidance(p_t, c_t, team_util, agent_moved)
        elif self._active:
            if self._phase == "avoidance":
                self._phase = "program_haul"
                self._unmoved_steps = 0
                self._accumulate(p_t, c_t, team_util, agent_moved)
            else:
                self._accumulate(p_t, c_t, team_util, agent_moved)

        return closed

    def _open_avoidance(self, p_t: float, c_t: float, team_util: float, agent_moved: bool):
        self._active = True
        self._phase = "avoidance"
        self._unmoved_steps = 0
        self._p = int(p_t)
        self._c = int(c_t)
        self._team_util = float(team_util)
        if self._max_conflict_steps > 0 and self._max_conflict_steps_type == "distance" and not agent_moved:
            self._unmoved_steps = 1

    def _accumulate(self, p_t: float, c_t: float, team_util: float, agent_moved: bool):
        self._p += int(p_t)
        self._c += int(c_t)
        self._team_util += float(team_util)
        if self._phase == "avoidance" and self._max_conflict_steps > 0 and self._max_conflict_steps_type == "distance":
            if agent_moved:
                self._unmoved_steps = 0
            else:
                self._unmoved_steps += 1

    def _should_split_avoidance(self, agent_moved: bool) -> bool:
        if self._max_conflict_steps <= 0:
            return False
        if self._max_conflict_steps_type == "time":
            return self._p + self._c >= self._max_conflict_steps
        next_unmoved = 0 if agent_moved else self._unmoved_steps + 1
        return next_unmoved >= self._max_conflict_steps

    def _split_avoidance(self, p_t: float, c_t: float, team_util: float, agent_moved: bool) -> ClosedConflictInstance:
        closed = self._close()
        self._open_avoidance(p_t, c_t, team_util, agent_moved)
        return closed

    def finalize_episode(self) -> ClosedConflictInstance | None:
        if not self._active:
            return None
        return self._close()

    def _close(self) -> ClosedConflictInstance:
        length = self._p + self._c
        if length <= 0:
            raise ValueError("conflict instance cannot close with zero length")
        inst = ClosedConflictInstance(
            p_norm=float(self._p) / float(length),
            c_norm=float(self._c) / float(length),
            team_util_norm=float(self._team_util) / float(length),
            length=length,
        )
        self._closed.append(inst)
        self._active = False
        self._phase = ""
        self._p = 0
        self._c = 0
        self._team_util = 0.0
        self._unmoved_steps = 0
        return inst

    def running_p_norm(self) -> float:
        length = self._p + self._c
        if length <= 0:
            return 0.0
        return float(self._p) / float(length)

    def running_c_norm(self) -> float:
        length = self._p + self._c
        if length <= 0:
            return 0.0
        return float(self._c) / float(length)

    def running_team_util_norm(self) -> float:
        length = self._p + self._c
        if length <= 0:
            return 0.0
        return float(self._team_util) / float(length)

    @property
    def active(self) -> bool:
        return self._active

    def mean_closed_p_norm(self) -> float:
        if len(self._closed) == 0:
            return 0.0
        return float(sum(inst.p_norm for inst in self._closed) / len(self._closed))

    def mean_closed_c_norm(self) -> float:
        if len(self._closed) == 0:
            return 0.0
        return float(sum(inst.c_norm for inst in self._closed) / len(self._closed))

    def mean_closed_team_util_norm(self) -> float:
        if len(self._closed) == 0:
            return 0.0
        return float(sum(inst.team_util_norm for inst in self._closed) / len(self._closed))


def new_conflict_trackers(
    agent_ids: list[str],
    max_conflict_steps: int,
    max_conflict_steps_type: str,
) -> dict[str, ConflictInstanceTracker]:
    return {
        agent_id: ConflictInstanceTracker(max_conflict_steps, max_conflict_steps_type)
        for agent_id in agent_ids
    }


def episode_mean_own_p_norm(trackers: dict[str, ConflictInstanceTracker], agent_ids: list[str]) -> float:
    return float(sum(trackers[agent_id].mean_closed_p_norm() for agent_id in agent_ids) / len(agent_ids))


def episode_mean_team_util_norm(trackers: dict[str, ConflictInstanceTracker], agent_ids: list[str]) -> float:
    values = [trackers[agent_id].mean_closed_team_util_norm() for agent_id in agent_ids]
    return float(sum(values) / len(values))


def episode_mean_own_p_norm_all_envs(
    trackers_by_env: list[dict[str, ConflictInstanceTracker]],
    agent_ids: list[str],
) -> float:
    return float(np.mean([episode_mean_own_p_norm(trackers, agent_ids) for trackers in trackers_by_env]))


def episode_mean_team_util_norm_all_envs(
    trackers_by_env: list[dict[str, ConflictInstanceTracker]],
    agent_ids: list[str],
) -> float:
    return float(np.mean([episode_mean_team_util_norm(trackers, agent_ids) for trackers in trackers_by_env]))
