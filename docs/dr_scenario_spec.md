# Forced DR Scenario Specification

This document translates the requested 3x3 scenario definitions into exact equations and code-facing terms.

## Core Quantities

- Let per-step team utility be:
  - `Gamma_obs = observed_util = mean_j (P_j - C_j)` (agent-mean, in `[-1, 1]` per step)
- Let agent `i` utility be:
  - `my_util = my_P - my_C`
- Let:
  - `full_duration = 1` be the maximum normalized utility for a full conflict window
    (`p+c` over the instance; with length-normalized `my_P` / `my_C`, a full program or
    avoidance window has share 1)
  - `my_mean` be a deterministic scalar (running mean of observed `P^i` samples)

For each scenario `(role, others_model)`, define:

- `team_util_without_me(role, others_model)`
- `team_util_with_me(role, others_model)`
- Scenario difference reward:
  - `D(role, others_model) = team_util_with_me(role, others_model) - team_util_without_me(role, others_model)`

## Axis Semantics

The two axes are separable:

- **Others axis** (`b`): governs `U_with` — what others are doing in the *known* current state (with me present). Since the others' behavior is observed but my individual role is indeterminate from their perspective, `U_with` depends only on this axis.
- **My-role axis** (`a`): governs `U_without` — the *counterfactual* baseline of what the team would do if I were removed. This depends only on my role, not on which others scenario is active.

## Forced Values (With-Me Component)

`U_with` is a pure function of the others axis:

- `*-AllC`: `team_util_with_me = 0`
- `*-AllP`: `team_util_with_me = full_duration`
- `*-AllSame`: `team_util_with_me = observed_util`

## Forced Values (Without-Me Component)

`U_without` is a pure function of the my-role axis:

- `solver-*`: `team_util_without_me = 0`
- `neutral-*`: `team_util_without_me = my_P`
- `causer-*`: `team_util_without_me = full_duration`

## Derived Scenario Rewards (All 9)

1. `solver-AllC`: `D = 0 - 0 = 0`
2. `solver-AllP`: `D = full_duration - 0 = full_duration`
3. `solver-AllSame`: `D = observed_util - 0 = observed_util`
4. `neutral-AllC`: `D = 0 - my_P = -my_P`
5. `neutral-AllP`: `D = full_duration - my_P`
6. `neutral-AllSame`: `D = observed_util - my_P`
7. `causer-AllC`: `D = 0 - full_duration = -full_duration`
8. `causer-AllP`: `D = full_duration - full_duration = 0`
9. `causer-AllSame`: `D = observed_util - full_duration`

## Math-to-Code Mapping Targets

- `Gamma_obs` / `observed_util` -> current `team_utility` batch/scalar passed to DR learner
- `my_P` -> per closed conflict instance, `sum(p_t) / (sum(p_t) + sum(c_t))` over the
  full episode of that conflict: avoidance steps while `n_local > 1`, then program
  steps (including immune haul) until `n_local > 1` again; averaged over closed instances.
  Still splits when `max_conflict_steps` is exceeded without leaving the neighborhood
  (forced close + new avoidance instance; current step only in the new window).
- `my_C` -> `1 - my_P` for the same instance (i.e. `sum(c_t) / (sum(p_t) + sum(c_t))`)
- `full_duration` -> `FULL_DURATION_NORM` (1.0) in `config.py`
- UCB bandit DR updates use `softplus(D)` so rewards are positive and stationary
- Default bandit credit (`instance_credited`): one UCB update per **closed conflict instance** for the active macro arm, using that instance's normalized `p`, `c`, and team util (not episode-shared)
- `my_mean` -> running mean of observed `P^i` samples (used in stats/logging only; no longer appears in scenario table)
- `D(role, others)` -> per-scenario reward tensor `scenario_delta = with_me - without_me`

## Notes For Implementation

- `U_with` is constant across all three role rows (same `[0, full_duration, observed_util]` for solver, neutral, causer).
- `U_without` is constant across all three others columns (same scalar repeated for each role).
- The gating mechanism `u(role|s)`, `v(others|s)`, and `w=u*v` remains unchanged.
- The mixed reward stays:
  - `R_mix = sum_{role,others} w(role,others|s) * D(role,others)`
