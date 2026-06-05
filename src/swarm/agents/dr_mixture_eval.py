"""
Read-only DR scenario-mixture forward pass for visualization / playback.
Mirrors DRScenarioMixtureLearner._forward_tensors (2x2 scenario grid).
"""

import json
from pathlib import Path

import numpy as np
import torch

from .networks import DRScenarioMixtureNet
from .device import DEVICE


class DRMixtureEvaluator:
    def __init__(
        self,
        state_dict_path: str | Path,
        full_duration: float,
        n_agents: int,
        my_mean: float,
        obs_dim: int,
        hidden_dim: int = 64,
    ):
        self.full_duration = float(full_duration)
        self.n_agents = int(n_agents)
        self.my_mean = float(my_mean)
        if self.full_duration <= 0.0:
            raise ValueError("full_duration must be positive")
        self.net = DRScenarioMixtureNet(obs_dim, hidden_dim).to(DEVICE)
        state_dict = torch.load(state_dict_path, map_location=DEVICE)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    @staticmethod
    def load_meta(meta_path: str | Path) -> dict:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    def evaluate(
        self,
        obs: np.ndarray,
        local_utility: float,
        own_p: float,
        own_c: float,
    ) -> dict[str, np.ndarray | float]:
        """
        obs: (obs_dim,) float32 or int32 (cast to float)
        Returns u, v (2,), w, scenario_delta (2,2), mixed_reward scalar.
        """
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            local_utility_t = torch.tensor([float(local_utility)], dtype=torch.float32, device=DEVICE)
            own_p_t = torch.tensor([float(own_p)], dtype=torch.float32, device=DEVICE)
            own_c_t = torch.tensor([float(own_c)], dtype=torch.float32, device=DEVICE)

            self_logits, others_logits = self.net(obs_t)
            u = torch.softmax(self_logits, dim=1)
            v = torch.softmax(others_logits, dim=1)
            w = u.unsqueeze(2) * v.unsqueeze(1)

            self_util = own_p_t - own_c_t
            fd_n = self.full_duration * float(self.n_agents - 1)
            solver_row = torch.stack((local_utility_t, local_utility_t + self_util), dim=1)
            causer_row = torch.stack((self_util, local_utility_t - fd_n), dim=1)
            scenario_delta = torch.stack((solver_row, causer_row), dim=1)
            mixed_reward = (w * scenario_delta).sum(dim=(1, 2))

            u_np = u.squeeze(0).cpu().numpy()
            v_np = v.squeeze(0).cpu().numpy()
            w_np = w.squeeze(0).cpu().numpy()
            delta_np = scenario_delta.squeeze(0).cpu().numpy()
            r_mix = float(mixed_reward.squeeze(0).item())

        return {
            "u": u_np,
            "v": v_np,
            "w": w_np,
            "scenario_delta": delta_np,
            "mixed_reward": r_mix,
        }
