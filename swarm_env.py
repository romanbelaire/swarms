"""Backward-compatible import surface for environment class."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm.env import RationalSwarmForagingEnv

__all__ = ["RationalSwarmForagingEnv"]
