from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class CriticReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, p_t: float, c_t: float, next_obs: np.ndarray):
        self.buffer.append((obs, p_t, c_t, next_obs))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, p_ts, c_ts, next_obs = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(p_ts, dtype=np.float32),
            np.array(c_ts, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class MetaPolicyReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, dr_idx: int, reward: float):
        self.buffer.append((obs, dr_idx, reward))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, dr_idxs, rewards = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(dr_idxs, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
