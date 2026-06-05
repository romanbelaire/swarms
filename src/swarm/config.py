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

# Conflict macros for bandit / baseline modes (canonical names).
ENABLED_CONFLICT_ARM_NAMES = (
    "freeze_tag",
    "randomwalk3",
    "wait3",
    "move_clear",
    "backwards2",
)
N_CONFLICT_ARMS = len(ENABLED_CONFLICT_ARM_NAMES)
BANDIT_CONFLICT_ARMS_CSV = ",".join(ENABLED_CONFLICT_ARM_NAMES)

# Learned task_avoid actions aligned with ENABLED_CONFLICT_ARM_NAMES (freeze_tag is conflict-only).
TASK_AVOID_ENABLED_ACTION_IDS = [
    NO_OP_3_TURNS_ACTION,
    MOVE_BACKWARDS_2_ACTION,
    RANDOM_WALK_3_ACTION,
    MOVE_CLEAR_ACTION,
]
N_ACTIONS_TASK_AVOID = len(TASK_AVOID_ENABLED_ACTION_IDS)

# Conflict-instance window cap for P/C normalization (0 = no cap).
MAX_CONFLICT_STEPS = 0
MAX_CONFLICT_STEPS_TYPE = "time"

# Max normalized conflict-window utility (p+c length-normalized P/C => full window = 1).
FULL_DURATION_NORM = 1.0

ENV_LAYOUT_CENTER_BASE = "center_base"
ENV_LAYOUT_DUAL_QUADRANT_BASE = "dual_quadrant_base"
ENV_LAYOUT_NAMES = (ENV_LAYOUT_CENTER_BASE, ENV_LAYOUT_DUAL_QUADRANT_BASE)
DEFAULT_ENV_LAYOUT = ENV_LAYOUT_CENTER_BASE
EXPERT_CHECKPOINT_DUAL_QUADRANT_BASE = "artifacts/expert/dqn_dual_quadrant_base_agent_0.pt"
ABLATION_RUN_NAME_SUFFIX_DUAL = "_dualqb"


def ablation_run_name_suffix(env_layout: str) -> str:
    if env_layout == ENV_LAYOUT_DUAL_QUADRANT_BASE:
        return ABLATION_RUN_NAME_SUFFIX_DUAL
    return ""

