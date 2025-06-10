import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from env import SmartIrrigationEnv

class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self):
        if self.locals.get("dones"):
            episode_reward = self.locals.get("rewards")
            if episode_reward is not None:
                self.episode_rewards.append(episode_reward)
        return True

env = SmartIrrigationEnv()
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    train_freq=4,
    target_update_interval=1000,
    verbose=1
)

reward_callback = RewardLoggerCallback()
model.learn(total_timesteps=200_000, callback=reward_callback)
model.save("models/irrigation_dqn_v2")

plt.figure(figsize=(10, 5))
plt.plot(reward_callback.episode_rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()