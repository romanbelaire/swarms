"""Shared configuration constants for swarm scripts."""

N_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 50
# Solo expert needs time to reach food and base; main DR train keeps short horizons by default.
EXPERT_DEFAULT_MAX_STEPS_PER_EPISODE = 400
EXPERT_EVAL_EVERY_EPISODES = 100
# Match common ablation CLI defaults (`run_ablations.py`); override per run as needed.
GRID_SIZE = 10
NUM_FOOD = 10
LOCAL_GRID_SIZE = 5
# Tiles, then planar position in [0, 2], carry {0, 1}.
# Freeze / immune durations live in infos. Immune agents see peers masked out in tile channels
# (empty ground) while keeping this shape so legacy experts still load.
OBS_DIM = LOCAL_GRID_SIZE * LOCAL_GRID_SIZE + 2 + 1

N_ACTIONS_FULL = 5

TASK_POLICY_ACTION = 0
NO_OP_3_TURNS_ACTION = 1
MOVE_BACKWARDS_2_ACTION = 2
RANDOM_WALK_3_ACTION = 3
WAIT2_THEN_FORWARD1_ACTION = 4
MOVE_CLEAR_ACTION = 5
HANDSHAKE_ACTION = 6
RESERVE_PARITY_ACTION = 7
RESERVE_PARITY_ESCAPE_ACTION = 8

N_CONFLICT_ARMS = 11  # len(swarm.training.train.CONFLICT_ACTION_NAMES)
CONFLICT_ARM_PRIORITY_SWAP_N3 = 8
CONFLICT_ARM_PASS_FOOD_N3 = 9
CONFLICT_ARM_FREEZE_TAG = 10

TASK_AVOID_ENABLED_ACTION_IDS = [
    action_id
    for action_id in (
        TASK_POLICY_ACTION,
        NO_OP_3_TURNS_ACTION,
        MOVE_BACKWARDS_2_ACTION,
        RANDOM_WALK_3_ACTION,
        WAIT2_THEN_FORWARD1_ACTION,
        MOVE_CLEAR_ACTION,
        HANDSHAKE_ACTION,
        RESERVE_PARITY_ACTION,
        RESERVE_PARITY_ESCAPE_ACTION,
    )
    if action_id >= 0
]
N_ACTIONS_TASK_AVOID = len(TASK_AVOID_ENABLED_ACTION_IDS)

# Conflict-instance window cap for P/C normalization (0 = no cap).
MAX_CONFLICT_STEPS = 0
MAX_CONFLICT_STEPS_TYPE = "time"

# Max normalized conflict-window utility (p+c length-normalized P/C => full window = 1).
FULL_DURATION_NORM = 1.0
