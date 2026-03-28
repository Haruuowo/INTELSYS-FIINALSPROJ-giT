"""
rl_agent.py
===========
RL Threshold-Tuning Agent Stub — Week 2
Manufacturing Defect Detection | INTELSYS AY2526

Design:
  - State  : recent false-alarm rate + confidence score distribution
  - Action : adjust detection threshold (lower / keep / raise)
  - Reward : -FP_rate + TP_rate  (penalize false alarms, reward true positives)

Status: STUBBED — reward function and environment defined.
        Full training loop to be completed in Week 3.

Author: De Castro, Juan Carlo C.
"""

import numpy as np

# ── Reward Function ──────────────────────────────────────────────────────────
def compute_reward(tp: int, fp: int, fn: int, tn: int,
                   fa_penalty: float = 2.0) -> float:
    """
    Reward = TPR - fa_penalty * FPR
    Heavily penalises false alarms (FPR) to meet ≤5% false-alarm target.
    """
    tpr = tp / (tp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    return tpr - fa_penalty * fpr


# ── Environment Stub ─────────────────────────────────────────────────────────
class ThresholdEnv:
    """
    Toy environment: agent adjusts threshold on a fixed score distribution.
    Real implementation will hook into CNN confidence outputs.
    """
    def __init__(self, scores: np.ndarray, labels: np.ndarray,
                 threshold: float = 0.5):
        self.scores    = scores
        self.labels    = labels
        self.threshold = threshold
        self.step_n    = 0
        self.max_steps = 20

    def _confusion(self):
        preds = (self.scores >= self.threshold).astype(int)
        tp = int(np.sum((preds == 1) & (self.labels == 1)))
        fp = int(np.sum((preds == 1) & (self.labels == 0)))
        fn = int(np.sum((preds == 0) & (self.labels == 1)))
        tn = int(np.sum((preds == 0) & (self.labels == 0)))
        return tp, fp, fn, tn

    def state(self):
        tp, fp, fn, tn = self._confusion()
        fpr = fp / (fp + tn + 1e-8)
        tpr = tp / (tp + fn + 1e-8)
        return np.array([self.threshold, fpr, tpr], dtype=np.float32)

    def step(self, action: int):
        """
        Actions: 0 = lower threshold (-0.05)
                 1 = keep threshold
                 2 = raise threshold (+0.05)
        """
        delta = {0: -0.05, 1: 0.0, 2: 0.05}[action]
        self.threshold = float(np.clip(self.threshold + delta, 0.1, 0.95))
        tp, fp, fn, tn = self._confusion()
        reward = compute_reward(tp, fp, fn, tn)
        self.step_n += 1
        done = self.step_n >= self.max_steps
        return self.state(), reward, done


# ── Agent Stub (random policy — to be replaced with DQN) ─────────────────────
class RandomAgent:
    """Placeholder random-policy agent. DQN to be implemented Week 3."""
    def select_action(self, state: np.ndarray) -> int:
        return np.random.randint(0, 3)


# ── Quick demo run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    # Simulate CNN confidence scores
    scores = np.concatenate([
        np.random.beta(8, 2, 60),   # defective (high score)
        np.random.beta(2, 8, 40),   # good (low score)
    ])
    labels = np.concatenate([np.ones(60), np.zeros(40)])

    env   = ThresholdEnv(scores, labels, threshold=0.5)
    agent = RandomAgent()

    print("RL Agent Stub — Early Learning Curve")
    print(f"{'Step':>5}  {'Threshold':>10}  {'Reward':>8}  {'Done':>5}")
    total_reward = 0
    state = env.state()
    for _ in range(20):
        action = agent.select_action(state)
        state, reward, done = env.step(action)
        total_reward += reward
        print(f"{env.step_n:>5}  {env.threshold:>10.3f}  {reward:>8.4f}  {str(done):>5}")
        if done:
            break
    print(f"\nTotal reward (random policy): {total_reward:.4f}")
    print("[NOTE] Full DQN training to be implemented in Week 3.")
