import random
import numpy as np
import torch
import torch.nn as nn

from .device import DEVICE
from .networks import QNetwork, PCCritic, DRScenarioMixtureNet
from .replay import ReplayBuffer, CriticReplayBuffer


def _is_cuda_overflow_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "cuda error" in message
        or "cublas" in message
        or "cudnn" in message
        or "allocator" in message
    )


def _safe_module_to_device(module: nn.Module, preferred_device: torch.device, model_label: str) -> tuple[nn.Module, torch.device]:
    try:
        return module.to(preferred_device), preferred_device
    except RuntimeError as exc:
        if preferred_device.type != "cuda" or not _is_cuda_overflow_error(exc):
            raise
        fallback_device = torch.device("cpu")
        print(f"[device-fallback] {model_label}: CUDA unavailable/overflow, using CPU")
        return module.to(fallback_device), fallback_device


def _safe_torch_load(checkpoint_path: str, preferred_device: torch.device, model_label: str):
    try:
        return torch.load(checkpoint_path, map_location=preferred_device)
    except RuntimeError as exc:
        if preferred_device.type != "cuda" or not _is_cuda_overflow_error(exc):
            raise
        fallback_device = torch.device("cpu")
        print(f"[device-fallback] {model_label}: checkpoint load moved to CPU")
        return torch.load(checkpoint_path, map_location=fallback_device)


class PCCriticLearner:
    def __init__(
        self,
        obs_dim: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        buffer_size: int = 25_000,
        batch_size: int = 64,
        target_update_freq: int = 3000,
        hidden_dim: int = 64,
    ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0
        self.net, self.device = _safe_module_to_device(PCCritic(obs_dim, hidden_dim), DEVICE, "PCCriticLearner.net")
        self.target_net, _ = _safe_module_to_device(PCCritic(obs_dim, hidden_dim), self.device, "PCCriticLearner.target_net")
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = CriticReplayBuffer(buffer_size)

    def store(self, obs: np.ndarray, p_t: float, c_t: float, next_obs: np.ndarray):
        self.buffer.push(obs, p_t, c_t, next_obs)

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        obs, p_ts, c_ts, next_obs = self.buffer.sample(self.batch_size)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        p_t_t = torch.tensor(p_ts, dtype=torch.float32, device=self.device)
        c_t_t = torch.tensor(c_ts, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        p_pred, c_pred = self.net(obs_t)
        with torch.no_grad():
            next_p, next_c = self.target_net(next_obs_t)
            g = self.gamma
            target_p = (1.0 - g) * p_t_t + g * next_p
            target_c = (1.0 - g) * c_t_t + g * next_c

        loss = nn.functional.mse_loss(p_pred, target_p) + nn.functional.mse_loss(c_pred, target_c)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        return float(loss.item())

    def predict_rates_batch(self, obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            x = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
            p, c = self.net(x)
            return p.cpu().numpy(), c.cpu().numpy()


class DRScenarioMixtureLearner:
    def __init__(
        self,
        obs_dim: int,
        lr: float = 5e-4,
        hidden_dim: int = 64,
        entropy_coef: float = 0.0,
        full_duration: float = 1.0,
        my_mean_mode: str = "running_own_p_mean",
    ):
        self.net, self.device = _safe_module_to_device(
            DRScenarioMixtureNet(obs_dim, hidden_dim), DEVICE, "DRScenarioMixtureLearner.net"
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.entropy_coef = entropy_coef
        self.full_duration = float(full_duration)
        self.my_mean_mode = my_mean_mode
        self.running_own_p_sum = 0.0
        self.running_own_p_count = 0
        if self.my_mean_mode != "running_own_p_mean":
            raise ValueError(f"Unsupported my_mean_mode: {self.my_mean_mode}")
        if self.full_duration <= 0.0:
            raise ValueError("full_duration must be positive")

    @staticmethod
    def _require_finite(x: torch.Tensor, name: str):
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains non-finite values")

    def _update_running_my_mean(self, own_p_t: torch.Tensor):
        own_p_cpu = own_p_t.detach().cpu().numpy()
        self.running_own_p_sum += float(own_p_cpu.sum())
        self.running_own_p_count += int(own_p_cpu.shape[0])
        if self.running_own_p_count <= 0:
            raise ValueError("running_own_p_count must stay positive after update")

    def _forward_tensors(self, obs_t: torch.Tensor, team_utility_t: torch.Tensor, own_p_t: torch.Tensor, own_c_t: torch.Tensor):
        self_logits, others_logits = self.net(obs_t)
        u = torch.softmax(self_logits, dim=1)
        v = torch.softmax(others_logits, dim=1)
        w = u.unsqueeze(2) * v.unsqueeze(1)
        own_utility = own_p_t - own_c_t
        my_mean_value = self.running_own_p_sum / self.running_own_p_count
        my_mean_t = torch.full_like(team_utility_t, float(my_mean_value))
        full_duration_t = torch.full_like(team_utility_t, float(self.full_duration))
        zero_t = torch.zeros_like(team_utility_t)

        # U_without: pure function of my-role axis (constant across others columns)
        # solver=0, neutral=my_P, causer=full_duration
        without_me = torch.stack(
            (
                torch.stack((zero_t, zero_t, zero_t), dim=1),
                torch.stack((own_p_t, own_p_t, own_p_t), dim=1),
                torch.stack((full_duration_t, full_duration_t, full_duration_t), dim=1),
            ),
            dim=1,
        )
        # U_with: pure function of others axis (identical across role rows)
        # AllC=0, AllP=full_duration, AllSame=obs_util
        with_me = torch.stack(
            (
                torch.stack((zero_t, full_duration_t, team_utility_t), dim=1),
                torch.stack((zero_t, full_duration_t, team_utility_t), dim=1),
                torch.stack((zero_t, full_duration_t, team_utility_t), dim=1),
            ),
            dim=1,
        )
        scenario_delta = with_me - without_me
        mixed_reward = (w * scenario_delta).sum(dim=(1, 2))
        self._require_finite(mixed_reward, "mixed_reward")
        return {"u": u, "v": v, "mixed_reward": mixed_reward, "own_utility": own_utility, "my_mean": my_mean_t}

    def reward_batch(self, obs_batch: np.ndarray, team_utility_batch: np.ndarray, own_p_batch: np.ndarray, own_c_batch: np.ndarray):
        with torch.no_grad():
            obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
            team_utility_t = torch.tensor(team_utility_batch, dtype=torch.float32, device=self.device)
            own_p_t = torch.tensor(own_p_batch, dtype=torch.float32, device=self.device)
            own_c_t = torch.tensor(own_c_batch, dtype=torch.float32, device=self.device)
            self._update_running_my_mean(own_p_t)
            out = self._forward_tensors(obs_t, team_utility_t, own_p_t, own_c_t)
            stats = {"u": out["u"].cpu().numpy(), "v": out["v"].cpu().numpy(), "mixed_reward": out["mixed_reward"].cpu().numpy()}
            return out["mixed_reward"].cpu().numpy(), stats

    def train_step_batch(self, obs_batch: np.ndarray, team_utility_batch: np.ndarray, own_p_batch: np.ndarray, own_c_batch: np.ndarray):
        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        team_utility_t = torch.tensor(team_utility_batch, dtype=torch.float32, device=self.device)
        own_p_t = torch.tensor(own_p_batch, dtype=torch.float32, device=self.device)
        own_c_t = torch.tensor(own_c_batch, dtype=torch.float32, device=self.device)
        out = self._forward_tensors(obs_t, team_utility_t, own_p_t, own_c_t)
        reward_loss = nn.functional.mse_loss(out["mixed_reward"], out["own_utility"])
        entropy_u = -(out["u"] * torch.log(out["u"] + 1e-12)).sum(dim=1).mean()
        entropy_v = -(out["v"] * torch.log(out["v"] + 1e-12)).sum(dim=1).mean()
        loss = reward_loss - self.entropy_coef * (entropy_u + entropy_v)
        self._require_finite(loss, "dr_mixture_loss")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "u_solver": float(out["u"][:, 0].mean().item()),
            "u_neutral": float(out["u"][:, 1].mean().item()),
            "u_causer": float(out["u"][:, 2].mean().item()),
            "v_c": float(out["v"][:, 0].mean().item()),
            "v_p": float(out["v"][:, 1].mean().item()),
            "v_mean": float(out["v"][:, 2].mean().item()),
            "mixed_reward": float(out["mixed_reward"].mean().item()),
            "my_mean": float(out["my_mean"].mean().item()),
        }


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 1000,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        hidden_dim: int = 64,
        use_double_dqn: bool = False,
        use_huber: bool = False,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_huber = use_huber
        self.steps = 0
        self.epsilon_episodes = 0
        self.q_net, self.device = _safe_module_to_device(QNetwork(obs_dim, n_actions, hidden_dim), DEVICE, "DQNAgent.q_net")
        self.target_net, _ = _safe_module_to_device(
            QNetwork(obs_dim, n_actions, hidden_dim), self.device, "DQNAgent.target_net"
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def epsilon(self) -> float:
        if self.epsilon_episodes >= self.epsilon_decay_steps:
            return self.epsilon_end
        frac = self.epsilon_episodes / self.epsilon_decay_steps
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def advance_epsilon_episode(self):
        self.epsilon_episodes += 1

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon():
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(x)
            return int(q.argmax(dim=1).item())

    def select_actions_batch(self, obs_batch: np.ndarray, deterministic: bool = False) -> np.ndarray:
        batch_size = len(obs_batch)
        eps = self.epsilon()
        with torch.no_grad():
            x = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
            greedy_actions = self.q_net(x).argmax(dim=1).cpu().numpy().astype(np.int64)
        if deterministic:
            return greedy_actions
        random_actions = np.random.randint(0, self.n_actions, size=batch_size, dtype=np.int64)
        explore_mask = np.random.random(size=batch_size) < eps
        return np.where(explore_mask, random_actions, greedy_actions)

    def store(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.buffer.push(obs, action, reward, next_obs, done)

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        q_values = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_net(next_obs_t).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs_t).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_obs_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)
        if self.use_huber:
            loss = nn.functional.smooth_l1_loss(q_values, targets)
        else:
            loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())


class FrozenTaskExpert:
    def __init__(self, obs_dim: int, checkpoint_path: str, n_actions: int = 5, hidden_dim: int = 64):
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for FrozenTaskExpert")
        self.q_net, self.device = _safe_module_to_device(
            QNetwork(obs_dim, n_actions, hidden_dim), DEVICE, "FrozenTaskExpert.q_net"
        )
        state_dict = _safe_torch_load(checkpoint_path, self.device, "FrozenTaskExpert")
        self.q_net.load_state_dict(state_dict)
        self.q_net.eval()

    def select_actions_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
            q = self.q_net(x)
            return q.argmax(dim=1).cpu().numpy().astype(np.int64)


class UCB1Bandit:
    """
    Non-contextual UCB1 bandit over conflict-action macro arms.
    """

    def __init__(self, n_arms: int):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.values = np.zeros(n_arms, dtype=np.float64)
        self.total_pulls = 0

    def select_arm(self) -> int:
        # Pull each arm once before UCB scoring.
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        exploration = np.sqrt(2.0 * np.log(float(self.total_pulls)) / self.counts.astype(np.float64))
        scores = self.values + exploration
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float):
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"arm index out of range: {arm}")
        self.total_pulls += 1
        self.counts[arm] += 1
        n = float(self.counts[arm])
        old = self.values[arm]
        self.values[arm] = old + (float(reward) - old) / n

    def arm_probabilities(self) -> np.ndarray:
        if self.total_pulls == 0:
            return np.zeros(self.n_arms, dtype=np.float64)
        return self.counts.astype(np.float64) / float(self.total_pulls)

