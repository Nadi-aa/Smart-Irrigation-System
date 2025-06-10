import gym
import numpy as np
from gym import spaces

class SmartIrrigationEnv(gym.Env):
    def __init__(self):
        super(SmartIrrigationEnv, self).__init__()

        low_obs = np.array([0.0, 0.0, 0.0, 0.0])
        high_obs = np.array([100.0, 2.0, 23.0, 2.0])
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Action space: 0=None, 1=Low, 2=Medium, 3=High
        self.action_space = spaces.Discrete(4)

        self.reset()

    def reset(self):
        self.soil_moisture = np.random.uniform(40, 70)
        self.weather = np.random.randint(0, 3)  # 0=sunny, 1=cloudy, 2=rainy
        self.time_of_day = np.random.randint(6, 18)
        self.plant_type = np.random.randint(0, 3)
        self.day_counter = 0
        self.done = False
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action)

        # Action effects on moisture
        irrigation_effects = [0, 5, 10, 20]
        self.soil_moisture += irrigation_effects[action]

        # Environmental evaporation effect
        evap_factor = [3.0, 1.5, 0.2]  # sunny > cloudy > rainy
        evap = evap_factor[self.weather] * (1 + self.plant_type * 0.5)
        self.soil_moisture -= evap

        # Clamp soil moisture
        self.soil_moisture = np.clip(self.soil_moisture, 0, 100)

        # Reward: optimal range depends on plant type
        optimal_ranges = [(50, 70), (60, 80), (70, 90)]
        low, high = optimal_ranges[self.plant_type]
        target = (low + high) / 2

        moisture_penalty = -abs(self.soil_moisture - target) / 50.0
        irrigation_cost = -0.01 * action  # penalize heavier irrigation
        reward = moisture_penalty + irrigation_cost

        # Update time
        self.time_of_day = (self.time_of_day + 1) % 24
        if self.time_of_day == 0:
            self.day_counter += 1

        # Occasionally change weather and plant
        if self.time_of_day % 6 == 0:
            self.weather = np.random.randint(0, 3)
        if self.time_of_day % 6 == 0:
            self.plant_type = (self.plant_type + 1) % 3

        # End after 7 days
        self.done = self.day_counter >= 7
        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        return np.array([
            self.soil_moisture,
            self.weather,
            self.time_of_day,
            self.plant_type
        ], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Time: {self.time_of_day}, Moisture: {self.soil_moisture:.2f}, Weather: {self.weather}, Plant: {self.plant_type}")
