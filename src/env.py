"""
env.py - Reinforcement Learning Environment for Multi-Source Log Correlation
MS Thesis: Multi-Source Heterogeneous Log Correlation using Reinforcement Learning

Custom Gymnasium Environment wrapping sequential multi-source log events.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


class LogCorrelationEnv(gym.Env):
    """
    Sequential Decision-Making Environment for Multi-Source Log Correlation.
    
    Observation:
      - Continuous normalized feature vector of length D (source_type, time_delta, IPs, ports, rules, etc.)
    
    Action:
      - Discrete(2):
          0 = Benign (Pass / Ignore event)
          1 = Attack / Correlate (Trigger alert / link event to active campaign)
          
    Reward Function:
      - True Positive  (Action=1, Target=1): +2.5
      - True Negative  (Action=0, Target=0): +0.1
      - False Positive (Action=1, Target=0): -0.5 (Penalty for false alarms)
      - False Negative (Action=0, Target=1): -3.0 (Severe penalty for missing attack stages)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        reward_tp: float = 2.5,
        reward_tn: float = 0.1,
        reward_fp: float = -0.5,
        reward_fn: float = -3.0
    ):
        super().__init__()
        assert len(features) == len(targets), "Features and targets must have the same length"
        
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.int64)
        self.num_samples = len(self.features)
        self.feature_dim = self.features.shape[1]

        self.reward_tp = reward_tp
        self.reward_tn = reward_tn
        self.reward_fp = reward_fp
        self.reward_fn = reward_fn

        # Action: 0 = Benign, 1 = Attack/Correlate
        self.action_space = spaces.Discrete(2)

        # Observation space: bounded [0, 1] normalized features
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.feature_dim,),
            dtype=np.float32
        )

        self.current_idx = 0
        self.total_reward = 0.0
        self.counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_idx = 0
        self.total_reward = 0.0
        self.counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

        obs = self.features[self.current_idx]
        info = {"current_idx": self.current_idx, "counts": self.counts}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        target = self.targets[self.current_idx]

        # Calculate reward
        if action == 1 and target == 1:
            reward = self.reward_tp
            self.counts["tp"] += 1
        elif action == 0 and target == 0:
            reward = self.reward_tn
            self.counts["tn"] += 1
        elif action == 1 and target == 0:
            reward = self.reward_fp
            self.counts["fp"] += 1
        else:  # action == 0 and target == 1
            reward = self.reward_fn
            self.counts["fn"] += 1

        self.total_reward += reward
        self.current_idx += 1

        terminated = self.current_idx >= self.num_samples
        truncated = False

        if not terminated:
            next_obs = self.features[self.current_idx]
        else:
            next_obs = np.zeros(self.feature_dim, dtype=np.float32)

        info = {
            "current_idx": self.current_idx,
            "counts": self.counts,
            "total_reward": self.total_reward
        }

        return next_obs, reward, terminated, truncated, info

    def render(self) -> None:
        pass
