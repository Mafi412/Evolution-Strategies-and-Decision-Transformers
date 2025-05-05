import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np


class DecisionTransformerHumanoidWrapper(gym.Wrapper):
    def __init__(self, env, max_length, desired_rtg, scale=1000.):
        super().__init__(env)
        self.max_length = max_length
        self.desired_rtg = desired_rtg / scale
        
        self._set_desired_shapes()
        
        self.observation_space = gym.spaces.Dict({
            "states": gym.spaces.Box(low=np.array([env.observation_space.low for _ in range(max_length)]), high=np.array([env.observation_space.high for _ in range(max_length)]), shape=self.state_shape, dtype=env.observation_space.dtype),
            "actions": gym.spaces.Box(low=np.array([env.action_space.low for _ in range(max_length)]), high=np.array([env.action_space.high for _ in range(max_length)]), shape=self.action_shape, dtype=env.action_space.dtype),
            "returns_to_go": gym.spaces.Box(low=np.array([[-float("inf")] for _ in range(max_length)]), high=np.array([[float("inf")] for _ in range(max_length)]), shape=self.rtg_shape, dtype=np.float32),
            "timesteps": gym.spaces.Box(low=np.array([0 for _ in range(max_length)]), high=np.array([999 for _ in range(max_length)]), shape=self.timesteps_shape, dtype=np.int32),
            "attention_mask": gym.spaces.Box(low=np.array([0 for _ in range(max_length)]), high=np.array([1 for _ in range(max_length)]), shape=self.attention_mask_shape, dtype=np.int32),
        })
        
        self._reset_data()
        self._set_current_state()
        
        
    def _set_desired_shapes(self):
        self.state_shape = (self.max_length,) + self.env.observation_space.shape
        self.action_shape = (self.max_length,) + self.env.action_space.shape
        self.rtg_shape = (self.max_length, 1)
        self.timesteps_shape = (self.max_length,)
        self.attention_mask_shape = (self.max_length,)
        
        
    def _reset_data(self):
        self.states = list(np.zeros(self.state_shape, dtype=self.env.observation_space.dtype))
        self.actions = list(np.zeros(self.action_shape, dtype=self.env.action_space.dtype))
        self.returns_to_go = list(np.zeros(self.rtg_shape, dtype=np.float32))
        self.timesteps = list(np.zeros(self.timesteps_shape, dtype=np.int32))
        self.attention_mask = list(np.zeros(self.attention_mask_shape, dtype=np.int32))
        

    def _set_current_state(self):
        self.current_state = dict()
        self.current_state["states"] = np.array(self.states[-self.max_length:], dtype=self.env.observation_space.dtype)
        self.current_state["actions"] = np.array(self.actions[-self.max_length:], dtype=self.env.action_space.dtype)
        self.current_state["returns_to_go"] = np.array(self.returns_to_go[-self.max_length:], dtype=np.float32)
        self.current_state["timesteps"] = np.array(self.timesteps[-self.max_length:], dtype=np.int32)
        self.current_state["attention_mask"] = np.array(self.attention_mask[-self.max_length:], dtype=np.int32)
    

    def reset(self, **kwargs):
        self._reset_data()
        
        observation, info = self.env.reset(**kwargs)
        self.states[-1] = observation
        self.returns_to_go[-1] = [self.desired_rtg]
        self.attention_mask[-1] = 1
        
        self._set_current_state()
        
        return self.current_state, info
    

    def step(self, action):
        self.actions[-1] = action
        self.actions.append(np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype))
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = reward / 1000.  # Scale reward to match the desired return-to-go
        
        self.states.append(observation)
        self.returns_to_go.append([self.returns_to_go[-1][0] - reward])
        self.timesteps.append(self.timesteps[-1] + 1 if self.timesteps[-1] < 999 else self.timesteps[-1]) # NOTE: The upper bound on the 999 timesteps instead of 1000 is due to some weird internal logic of Stable baselines, which tries to force the DT model to choose action even for the ended episode, for timestep >= 1000 (for which the embedding in the model is not defined), we just set the timestep to 999.
        self.attention_mask.append(1)
        
        self._set_current_state()
        
        return self.current_state, reward, terminated, truncated, info


def get_flattened_dt_humanoid_env(max_length, desired_rtg):
    return FlattenObservation(DecisionTransformerHumanoidWrapper(gym.make("Humanoid-v4", render_mode=None), max_length, desired_rtg))
