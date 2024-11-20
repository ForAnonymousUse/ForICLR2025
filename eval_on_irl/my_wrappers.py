from gymnasium import ObservationWrapper, ActionWrapper, Env, spaces
from sklearn.preprocessing import MinMaxScaler
from typing import TypeVar, Dict, Union, List
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

from vvcgym.env import VVCGym
from vvcgym.tasks.velocity_vector_control_task import VelocityVectorControlTask
from vvcgym.planes.f16_plane import F16Plane

PROJECT_ROOT_DIR = Path(__file__).parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

def get_min_max_scalar(mins: np.ndarray, maxs: np.ndarray, feature_range: Tuple[int, int]=(0., 1.)):
    scalar = MinMaxScaler(feature_range=feature_range, clip=True, copy=True)
    return scalar.fit([mins, maxs])

# from gymnasium core.py
ObsType = TypeVar("ObsType")
WrapperObsType = TypeVar("WrapperObsType")
ActType = TypeVar("ActType")
WrapperActType = TypeVar("WrapperActType")


class ScaledObservationWrapper(ObservationWrapper):

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

        plane_state_mins = VelocityVectorControlTask.get_state_lower_bounds()
        plane_state_maxs = VelocityVectorControlTask.get_state_higher_bounds()
        plane_goal_mins = VelocityVectorControlTask.get_goal_lower_bounds()
        plane_goal_maxs = VelocityVectorControlTask.get_goal_higher_bounds()

        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(low=0., high=1., shape=(len(plane_state_mins),)),
                # phi, theta, psi, v, mu, chi, p, h
                desired_goal=spaces.Box(low=0., high=1., shape=(len(plane_goal_mins),)),
                achieved_goal=spaces.Box(low=0., high=1., shape=(len(plane_goal_mins),)),
            )
        )

        self.state_scalar: MinMaxScaler = get_min_max_scalar(
            mins=np.array(plane_state_mins),
            maxs=np.array(plane_state_maxs),
            feature_range=(0., 1.),
        )
        self.goal_scalar: MinMaxScaler = get_min_max_scalar(
            mins=np.array(plane_goal_mins),
            maxs=np.array(plane_goal_maxs),
            feature_range=(0., 1.)
        )

    def scale_state(self, state_var: Union[Dict, np.ndarray]) -> Union[Dict, np.ndarray]:
        if isinstance(state_var, dict):
            tmp_state_var = [state_var]
            # return self.state_scalar.transform(tmp_state_var).reshape((-1))
        elif len(state_var.shape) == 2:
            tmp_state_var = state_var
            # return self.state_scalar.transform(state_var)
        else:
            raise TypeError("state_var only one or two dimension")

        res = [
            dict(
                observation=self.state_scalar.transform(tmp_state["observation"].reshape((1, -1))).reshape((-1)),
                desired_goal=self.goal_scalar.transform(tmp_state["desired_goal"].reshape((1, -1))).reshape((-1)),
                achieved_goal=self.goal_scalar.transform(tmp_state["achieved_goal"].reshape((1, -1))).reshape((-1)),
            )
            for tmp_state in tmp_state_var
        ]

        if isinstance(state_var, dict):
            return res[0]
        else:
            return np.array(res)

    def observation(self, observation: ObsType) -> WrapperObsType:
        return self.scale_state(observation)

    def inverse_scale_state(self, state_var: Union[Dict, np.ndarray]) -> Union[Dict, np.ndarray]:

        if isinstance(state_var, dict):
            tmp_state_var = [state_var]
            # return self.state_scalar.inverse_transform(tmp_state_var).reshape((-1))
        elif len(state_var.shape) == 2:
            tmp_state_var = state_var
            # return self.state_scalar.inverse_transform(state_var)
        else:
            raise TypeError("state_var only one or two dimension!")

        res = [
            dict(
                observation=self.state_scalar.inverse_transform(tmp_state["observation"].reshape((1, -1))).reshape(
                    (-1)),
                desired_goal=self.goal_scalar.inverse_transform(tmp_state["desired_goal"].reshape((1, -1))).reshape(
                    (-1)),
                achieved_goal=self.goal_scalar.inverse_transform(tmp_state["achieved_goal"].reshape((1, -1))).reshape(
                    (-1)),
            )
            for tmp_state in tmp_state_var
        ]

        if isinstance(state_var, dict):
            return res[0]
        else:
            return np.array(res)


class ScaledActionWrapper(ActionWrapper):

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

        action_mins = F16Plane.get_action_lower_bounds(env.unwrapped.plane.control_mode)
        action_maxs = F16Plane.get_action_higher_bounds(env.unwrapped.plane.control_mode)

        self.action_space = spaces.Box(low=0., high=1., shape=(len(action_mins),))  # p, nz, pla

 
        self.action_scalar: MinMaxScaler = get_min_max_scalar(
            mins=np.array(action_mins),
            maxs=np.array(action_maxs),
            feature_range=(0., 1.)
        )

    def inverse_scale_action(self, action_var: np.ndarray) -> np.ndarray:

        if len(action_var.shape) == 1:
            tmp_action_var = action_var.reshape((1, -1))
            return self.action_scalar.inverse_transform(tmp_action_var).reshape((-1))
        elif len(action_var.shape) == 2:
            return self.action_scalar.inverse_transform(action_var)
        else:
            raise TypeError("action_var only one or two dimension")

    def action(self, action: WrapperActType) -> ActType:

        if type(action) == np.ndarray:
            return self.inverse_scale_action(action)
        else:
            return self.inverse_scale_action(np.array(action))

    def scale_action(self, action_var: np.ndarray) -> np.ndarray:

        if len(action_var.shape) == 1:
            tmp_action_var = action_var.reshape((1, -1))
            return self.action_scalar.transform(tmp_action_var).reshape((-1))
        elif len(action_var.shape) == 2:
            return self.action_scalar.transform(action_var)
        else:
            raise TypeError("action_var only one or two dimension")