"""Backward-compatible import surface for agent classes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm.agents import DQNAgent, PCCriticLearner, DRScenarioMixtureLearner, FrozenTaskExpert, DEVICE

__all__ = ["DQNAgent", "PCCriticLearner", "DRScenarioMixtureLearner", "FrozenTaskExpert", "DEVICE"]
