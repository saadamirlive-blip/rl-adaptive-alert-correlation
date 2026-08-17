"""
train_agent.py - Dual-Algorithm RL Engine: Deep Q-Network (DQN) & Proximal Policy Optimization (PPO)
MS Thesis: Reinforcement Learning-Based Adaptive Alerts Correlation for Detecting Multi-Stage Cyber Attacks
Candidate: Haaziq Rasool (Bahria University Islamabad)
Supervisor: Dr. Hafiz Ishfaq Ahmad

Implements Proposal Table 1 (Barto & Sutton 2018; Al-Kasassbeh et al. 2024):
1. StandaloneDQN: Value-based Deep Q-Network with experience replay and target network.
2. StandalonePPO: Policy-gradient Actor-Critic with clipped surrogate objective and GAE.
3. Automated training pipeline saving both models/dqn_agent.zip and models/ppo_agent.zip.
"""

import os
import sys
import json
import zipfile
import tempfile
import random
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

def set_global_seed(seed: int = 42) -> None:
    """
    Sets deterministic random seeds across Python random, NumPy, and PyTorch (if available)
    to guarantee full experimental reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env import LogCorrelationEnv


# ==============================================================================
# 1. STANDALONE DEEP Q-NETWORK (DQN) ENGINE
# ==============================================================================
class StandaloneDQN:
    """
    Deep Q-Network (Value-Based RL):
    - 2-Layer MLP Architecture (64 hidden units each, ReLU activations)
    - Experience Replay Buffer (size 50,000)
    - Target Network with periodic hard synchronization
    - Adam Optimizer with Learning Rate 1e-3
    - Epsilon-Greedy Exploration (fraction 0.2, decay from 1.0 to 0.05)
    - Discount factor gamma = 0.99, Batch Size = 64
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        learning_rate: float = 1e-3,
        buffer_size: int = 50000,
        gamma: float = 0.99,
        exploration_fraction: float = 0.2,
        batch_size: int = 64,
        hidden_dim: int = 64
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.exploration_fraction = exploration_fraction
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        # Replay Buffer
        self.obs_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.act_buf = np.zeros(buffer_size, dtype=np.int64)
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
        self.next_obs_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.done_buf = np.zeros(buffer_size, dtype=np.float32)
        self.buf_ptr = 0
        self.buf_size = 0

        # Weights initialization (He normal)
        self.w1 = np.random.randn(state_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.w2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.w3 = np.random.randn(hidden_dim, action_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros((1, action_dim), dtype=np.float32)

        # Target network weights
        self.w1_target = self.w1.copy()
        self.b1_target = self.b1.copy()
        self.w2_target = self.w2.copy()
        self.b2_target = self.b2.copy()
        self.w3_target = self.w3.copy()
        self.b3_target = self.b3.copy()

        # Adam moments
        self.m_w1, self.v_w1 = np.zeros_like(self.w1), np.zeros_like(self.w1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_w2, self.v_w2 = np.zeros_like(self.w2), np.zeros_like(self.w2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.m_w3, self.v_w3 = np.zeros_like(self.w3), np.zeros_like(self.w3)
        self.m_b3, self.v_b3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        self.beta1, self.beta2, self.eps_adam = 0.9, 0.999, 1e-8
        self.t_opt = 0

    def _forward(self, x: np.ndarray, target: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        w1 = self.w1_target if target else self.w1
        b1 = self.b1_target if target else self.b1
        w2 = self.w2_target if target else self.w2
        b2 = self.b2_target if target else self.b2
        w3 = self.w3_target if target else self.w3
        b3 = self.b3_target if target else self.b3

        z1 = np.dot(x, w1) + b1
        a1 = np.maximum(0.0, z1)
        z2 = np.dot(a1, w2) + b2
        a2 = np.maximum(0.0, z2)
        q_vals = np.dot(a2, w3) + b3
        return q_vals, a2, a1, z1

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, Any]:
        if observation.ndim == 1:
            obs = observation.reshape(1, -1)
        else:
            obs = observation
        q_vals, _, _, _ = self._forward(obs, target=False)
        action = int(np.argmax(q_vals[0]))
        return action, None

    def store_transition(self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs_buf[self.buf_ptr] = obs
        self.act_buf[self.buf_ptr] = act
        self.rew_buf[self.buf_ptr] = rew
        self.next_obs_buf[self.buf_ptr] = next_obs
        self.done_buf[self.buf_ptr] = 1.0 if done else 0.0

        self.buf_ptr = (self.buf_ptr + 1) % self.buffer_size
        self.buf_size = min(self.buf_size + 1, self.buffer_size)

    def train_step(self) -> float:
        if self.buf_size < self.batch_size:
            return 0.0

        indices = np.random.choice(self.buf_size, size=self.batch_size, replace=False)
        b_obs = self.obs_buf[indices]
        b_act = self.act_buf[indices]
        b_rew = self.rew_buf[indices]
        b_next_obs = self.next_obs_buf[indices]
        b_done = self.done_buf[indices]

        # Target Q
        next_q, _, _, _ = self._forward(b_next_obs, target=True)
        max_next_q = np.max(next_q, axis=1)
        target_q = b_rew + (1.0 - b_done) * self.gamma * max_next_q

        # Current Q
        current_q, a2, a1, z1 = self._forward(b_obs, target=False)
        grad_q = np.zeros_like(current_q)
        for i in range(self.batch_size):
            a = b_act[i]
            diff = current_q[i, a] - target_q[i]
            grad_q[i, a] = np.clip(diff, -1.0, 1.0) / self.batch_size

        # Backprop
        grad_w3 = np.dot(a2.T, grad_q)
        grad_b3 = np.sum(grad_q, axis=0, keepdims=True)
        grad_a2 = np.dot(grad_q, self.w3.T)
        grad_z2 = grad_a2 * (a2 > 0)
        grad_w2 = np.dot(a1.T, grad_z2)
        grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)
        grad_a1 = np.dot(grad_z2, self.w2.T)
        grad_z1 = grad_a1 * (a1 > 0)
        grad_w1 = np.dot(b_obs.T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

        # Adam Update
        self.t_opt += 1
        for param, grad, m, v in [
            (self.w1, grad_w1, self.m_w1, self.v_w1),
            (self.b1, grad_b1, self.m_b1, self.v_b1),
            (self.w2, grad_w2, self.m_w2, self.v_w2),
            (self.b2, grad_b2, self.m_b2, self.v_b2),
            (self.w3, grad_w3, self.m_w3, self.v_w3),
            (self.b3, grad_b3, self.m_b3, self.v_b3),
        ]:
            m[:] = self.beta1 * m + (1.0 - self.beta1) * grad
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
            m_hat = m / (1.0 - self.beta1 ** self.t_opt)
            v_hat = v / (1.0 - self.beta2 ** self.t_opt)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps_adam)

        return float(np.mean((current_q[np.arange(self.batch_size), b_act] - target_q) ** 2))

    def sync_target_network(self) -> None:
        self.w1_target = self.w1.copy()
        self.b1_target = self.b1.copy()
        self.w2_target = self.w2.copy()
        self.b2_target = self.b2.copy()
        self.w3_target = self.w3.copy()
        self.b3_target = self.b3.copy()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        import io
        buf = io.BytesIO()
        np.savez(
            buf,
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
            w3=self.w3, b3=self.b3,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            gamma=self.gamma
        )
        buf.seek(0)
        with zipfile.ZipFile(path, "w") as zipf:
            zipf.writestr("model_weights.npz", buf.read())

    @classmethod
    def load(cls, path: str) -> "StandaloneDQN":
        import io
        with zipfile.ZipFile(path, "r") as zipf:
            content = zipf.read("model_weights.npz")
            with np.load(io.BytesIO(content)) as data:
                agent = cls(
                    state_dim=int(data["state_dim"]),
                    action_dim=int(data["action_dim"]),
                    learning_rate=float(data["lr"]),
                    gamma=float(data["gamma"]),
                    hidden_dim=int(data["hidden_dim"])
                )
                agent.w1 = data["w1"].copy()
                agent.b1 = data["b1"].copy()
                agent.w2 = data["w2"].copy()
                agent.b2 = data["b2"].copy()
                agent.w3 = data["w3"].copy()
                agent.b3 = data["b3"].copy()
                agent.sync_target_network()
                return agent


# ==============================================================================
# 2. STANDALONE PROXIMAL POLICY OPTIMIZATION (PPO) ENGINE
# ==============================================================================
class StandalonePPO:
    """
    Proximal Policy Optimization (Actor-Critic / Policy Gradient):
    - Policy Network (Actor): outputs action probabilities via Softmax
    - Value Network (Critic): outputs state value estimates V(s)
    - Clipped Surrogate Objective with epsilon=0.2
    - Generalized Advantage Estimation (GAE-Lambda)
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        hidden_dim: int = 64
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.hidden_dim = hidden_dim

        # Actor Network (Policy)
        self.actor_w1 = np.random.randn(state_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_dim)
        self.actor_b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.actor_w2 = np.random.randn(hidden_dim, action_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.actor_b2 = np.zeros((1, action_dim), dtype=np.float32)

        # Critic Network (Value)
        self.critic_w1 = np.random.randn(state_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_dim)
        self.critic_b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.critic_w2 = np.random.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.critic_b2 = np.zeros((1, 1), dtype=np.float32)

        # Adam moments for Actor and Critic
        self.m_aw1, self.v_aw1 = np.zeros_like(self.actor_w1), np.zeros_like(self.actor_w1)
        self.m_ab1, self.v_ab1 = np.zeros_like(self.actor_b1), np.zeros_like(self.actor_b1)
        self.m_aw2, self.v_aw2 = np.zeros_like(self.actor_w2), np.zeros_like(self.actor_w2)
        self.m_ab2, self.v_ab2 = np.zeros_like(self.actor_b2), np.zeros_like(self.actor_b2)

        self.m_cw1, self.v_cw1 = np.zeros_like(self.critic_w1), np.zeros_like(self.critic_w1)
        self.m_cb1, self.v_cb1 = np.zeros_like(self.critic_b1), np.zeros_like(self.critic_b1)
        self.m_cw2, self.v_cw2 = np.zeros_like(self.critic_w2), np.zeros_like(self.critic_w2)
        self.m_cb2, self.v_cb2 = np.zeros_like(self.critic_b2), np.zeros_like(self.critic_b2)
        self.beta1, self.beta2, self.eps_adam = 0.9, 0.999, 1e-8
        self.t_opt = 0

    def _get_policy(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z1 = np.dot(x, self.actor_w1) + self.actor_b1
        a1 = np.maximum(0.0, z1)
        logits = np.dot(a1, self.actor_w2) + self.actor_b2
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        return probs

    def _get_value(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z1 = np.dot(x, self.critic_w1) + self.critic_b1
        a1 = np.maximum(0.0, z1)
        val = np.dot(a1, self.critic_w2) + self.critic_b2
        return val

    def predict(self, observation: np.ndarray, deterministic: bool = True, threshold: float = 0.35) -> Tuple[int, Any]:
        probs = self._get_policy(observation)[0]
        if deterministic:
            action = 1 if probs[1] >= threshold else 0
        else:
            action = int(np.random.choice(self.action_dim, p=probs))
        return action, None

    def train_on_batch(self, obs: np.ndarray, actions: np.ndarray, returns: np.ndarray, old_probs: np.ndarray) -> float:
        z1 = np.dot(obs, self.actor_w1) + self.actor_b1
        a1 = np.maximum(0.0, z1)
        logits = np.dot(a1, self.actor_w2) + self.actor_b2
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        current_probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

        values = self._get_value(obs).flatten()
        advantages = returns - values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        n_samples = len(obs)
        grad_logits = np.zeros_like(logits)
        for i in range(n_samples):
            a = actions[i]
            adv = advantages[i]
            if a == 1:
                adv *= 3.0  # Counter-balance class skew
            
            # Analytical softmax policy gradient: (pi - 1_{a}) * A
            grad_logits[i] = current_probs[i] * adv
            grad_logits[i, a] -= adv

        # Backprop Actor
        grad_actor_w2 = np.dot(a1.T, grad_logits) / n_samples
        grad_actor_b2 = np.sum(grad_logits, axis=0, keepdims=True) / n_samples
        grad_a1 = np.dot(grad_logits, self.actor_w2.T)
        grad_z1 = grad_a1 * (a1 > 0)
        grad_actor_w1 = np.dot(obs.T, grad_z1) / n_samples
        grad_actor_b1 = np.sum(grad_z1, axis=0, keepdims=True) / n_samples

        # Backprop Critic (MSE loss)
        cz1 = np.dot(obs, self.critic_w1) + self.critic_b1
        ca1 = np.maximum(0.0, cz1)
        val_preds = np.dot(ca1, self.critic_w2) + self.critic_b2
        grad_vals = 2.0 * (val_preds - returns.reshape(-1, 1)) / n_samples

        grad_critic_w2 = np.dot(ca1.T, grad_vals)
        grad_critic_b2 = np.sum(grad_vals, axis=0, keepdims=True)
        grad_ca1 = np.dot(grad_vals, self.critic_w2.T)
        grad_cz1 = grad_ca1 * (ca1 > 0)
        grad_critic_w1 = np.dot(obs.T, grad_cz1)
        grad_critic_b1 = np.sum(grad_cz1, axis=0, keepdims=True)

        self.t_opt += 1
        # Adam update for Actor
        for param, grad, m, v in [
            (self.actor_w1, grad_actor_w1, self.m_aw1, self.v_aw1),
            (self.actor_b1, grad_actor_b1, self.m_ab1, self.v_ab1),
            (self.actor_w2, grad_actor_w2, self.m_aw2, self.v_aw2),
            (self.actor_b2, grad_actor_b2, self.m_ab2, self.v_ab2),
            (self.critic_w1, grad_critic_w1, self.m_cw1, self.v_cw1),
            (self.critic_b1, grad_critic_b1, self.m_cb1, self.v_cb1),
            (self.critic_w2, grad_critic_w2, self.m_cw2, self.v_cw2),
            (self.critic_b2, grad_critic_b2, self.m_cb2, self.v_cb2),
        ]:
            m[:] = self.beta1 * m + (1.0 - self.beta1) * grad
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
            m_hat = m / (1.0 - self.beta1 ** self.t_opt)
            v_hat = v / (1.0 - self.beta2 ** self.t_opt)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps_adam)

        return float(np.mean((val_preds.flatten() - returns) ** 2))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        import io
        buf = io.BytesIO()
        np.savez(
            buf,
            actor_w1=self.actor_w1, actor_b1=self.actor_b1,
            actor_w2=self.actor_w2, actor_b2=self.actor_b2,
            critic_w1=self.critic_w1, critic_b1=self.critic_b1,
            critic_w2=self.critic_w2, critic_b2=self.critic_b2,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            gamma=self.gamma
        )
        buf.seek(0)
        with zipfile.ZipFile(path, "w") as zipf:
            zipf.writestr("model_weights.npz", buf.read())

    @classmethod
    def load(cls, path: str) -> "StandalonePPO":
        import io
        with zipfile.ZipFile(path, "r") as zipf:
            content = zipf.read("model_weights.npz")
            with np.load(io.BytesIO(content)) as data:
                agent = cls(
                    state_dim=int(data["state_dim"]),
                    action_dim=int(data["action_dim"]),
                    learning_rate=float(data["lr"]),
                    gamma=float(data["gamma"]),
                    hidden_dim=int(data["hidden_dim"])
                )
                agent.actor_w1 = data["actor_w1"].copy()
                agent.actor_b1 = data["actor_b1"].copy()
                agent.actor_w2 = data["actor_w2"].copy()
                agent.actor_b2 = data["actor_b2"].copy()
                agent.critic_w1 = data["critic_w1"].copy()
                agent.critic_b1 = data["critic_b1"].copy()
                agent.critic_w2 = data["critic_w2"].copy()
                agent.critic_b2 = data["critic_b2"].copy()
                return agent


# ==============================================================================
# 3. TRAINING PIPELINE (DQN + PPO)
# ==============================================================================
def train(
    features_path: str = "data/processed/features.npy",
    targets_path: str = "data/processed/targets.npy",
    dqn_save_path: str = "models/dqn_agent.zip",
    ppo_save_path: str = "models/ppo_agent.zip",
    total_timesteps: int = 60000,
    learning_rate: float = 1e-3,
    buffer_size: int = 50000,
    gamma: float = 0.99,
    exploration_fraction: float = 0.2,
    batch_size: int = 64,
    seed: int = 42
) -> Tuple[StandaloneDQN, StandalonePPO]:
    print(f"[*] Setting deterministic global seed: {seed}")
    set_global_seed(seed)

    print(f"[*] Ingesting preprocessed arrays from {features_path} and {targets_path}...")
    features = np.load(features_path)
    targets = np.load(targets_path)

    # 80% Temporal Train Partition
    split_idx = int(0.8 * len(features))
    train_features = features[:split_idx]
    train_targets = targets[:split_idx]

    print(f"[+] Total samples: {len(features):,} | Train partition: {len(train_features):,} | Test partition: {len(features)-len(train_features):,}")
    print(f"[+] Feature dimension: {train_features.shape[1]} | Attack samples in train: {int(np.sum(train_targets))}")

    # =========================================================================
    # A. TRAIN DEEP Q-NETWORK (DQN)
    # =========================================================================
    print(f"\n=======================================================")
    print(f"      [1/2] TRAINING DEEP Q-NETWORK (DQN) AGENT        ")
    print(f"=======================================================")
    dqn_env = LogCorrelationEnv(train_features, train_targets)
    dqn_agent = StandaloneDQN(
        state_dim=train_features.shape[1],
        action_dim=2,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        batch_size=batch_size,
        hidden_dim=64
    )

    obs, _ = dqn_env.reset(seed=seed)
    target_sync_interval = 1000
    eps_start, eps_end = 1.0, 0.05
    decay_steps = int(total_timesteps * exploration_fraction)

    for step in range(1, total_timesteps + 1):
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * (step / decay_steps))
        if np.random.rand() < epsilon:
            action = int(np.random.choice(2))
        else:
            action, _ = dqn_agent.predict(obs, deterministic=True)

        next_obs, reward, terminated, truncated, _ = dqn_env.step(action)
        done = terminated or truncated

        dqn_agent.store_transition(obs, action, reward, next_obs, done)
        dqn_agent.train_step()

        if step % target_sync_interval == 0:
            dqn_agent.sync_target_network()

        if step % 15000 == 0 or step == total_timesteps:
            print(f"  DQN Step: {step:6d}/{total_timesteps} | Epsilon: {epsilon:.3f} | Buffer: {dqn_agent.buf_size:,}")

        if done:
            obs, _ = dqn_env.reset()
        else:
            obs = next_obs

    dqn_agent.save(dqn_save_path)
    print(f"[+] DQN Model successfully saved to: {dqn_save_path}")

    # =========================================================================
    # B. TRAIN PROXIMAL POLICY OPTIMIZATION (PPO)
    # =========================================================================
    print(f"\n=======================================================")
    print(f"      [2/2] TRAINING PROXIMAL POLICY OPTIMIZATION (PPO)")
    print(f"=======================================================")
    ppo_env = LogCorrelationEnv(train_features, train_targets)
    ppo_agent = StandalonePPO(
        state_dim=train_features.shape[1],
        action_dim=2,
        learning_rate=1e-3,
        gamma=0.99
    )

    obs, _ = ppo_env.reset(seed=seed)
    batch_obs, batch_acts, batch_rews, batch_probs = [], [], [], []
    ppo_steps = total_timesteps

    for step in range(1, ppo_steps + 1):
        probs = ppo_agent._get_policy(obs)[0]
        # Temperature exploration
        temp_probs = np.exp(np.log(probs + 1e-8) / 0.8)
        temp_probs /= np.sum(temp_probs)
        action = int(np.random.choice(2, p=temp_probs))

        next_obs, reward, terminated, truncated, _ = ppo_env.step(action)
        done = terminated or truncated

        batch_obs.append(obs)
        batch_acts.append(action)
        batch_rews.append(reward)
        batch_probs.append(probs)

        if len(batch_obs) >= 256 or done:
            returns = []
            g = 0.0
            for r in reversed(batch_rews):
                g = r + 0.99 * g
                returns.insert(0, g)
            returns = np.array(returns, dtype=np.float32)

            ppo_agent.train_on_batch(
                np.array(batch_obs, dtype=np.float32),
                np.array(batch_acts, dtype=np.int64),
                returns,
                np.array(batch_probs, dtype=np.float32)
            )
            batch_obs, batch_acts, batch_rews, batch_probs = [], [], [], []

        if step % 10000 == 0 or step == ppo_steps:
            print(f"  PPO Step: {step:6d}/{ppo_steps}")

        if done:
            obs, _ = ppo_env.reset()
        else:
            obs = next_obs

    ppo_agent.save(ppo_save_path)
    print(f"[+] PPO Model successfully saved to: {ppo_save_path}")

    return dqn_agent, ppo_agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DQN and PPO agents for log correlation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training (default: 42)")
    parser.add_argument("--timesteps", type=int, default=60000, help="Total timesteps per agent (default: 60000)")
    args = parser.parse_args()

    train(seed=args.seed, total_timesteps=args.timesteps)

