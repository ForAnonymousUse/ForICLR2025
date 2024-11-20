from typing import Tuple
import gymnasium
from gymnasium import spaces
from pathlib import Path
import numpy as np


class ConcatObsWrapper(gymnasium.ObservationWrapper):

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=np.array([-180., -90., -180., 0., -90., -180., -300., 0., 0., -90., -180.]),
            high=np.array([180., 90., 180., 1000., 90., 180., 300., 20000., 1000., 90., 180.]),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-180., -4., 0.]),
            high=np.array([180., 9., 1.]),
            dtype=np.float32
        )

    def observation(self, observation):
        return np.array([*observation["observation"], *observation["desired_goal"]])


