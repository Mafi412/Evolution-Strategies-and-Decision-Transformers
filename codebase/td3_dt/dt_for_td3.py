import torch
import numpy as np
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from components.decision_transformer.gym.models.decision_transformer import DecisionTransformer
import gymnasium as gym
    
    
def _transform_observation_batch_to_dict_of_batched_sequences(observation_space, observations):
    if isinstance(observations, torch.Tensor):
        observations = observations.cpu().numpy()
    unflattened_batch = [gym.spaces.unflatten(observation_space, flat_obs) for flat_obs in observations]
    return {k: np.stack([obs[k] for obs in unflattened_batch]) for k in unflattened_batch[0]}


class DecisionTransformerActor(torch.nn.Module):
    def __init__(self, dt_kwargs, observation_space):
        super().__init__()
        self.dt_kwargs = dt_kwargs
        self.observation_space = observation_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dt_model = DecisionTransformer(**dt_kwargs).to(self.device)


    def forward(self, observations):
        # For SB3, batch_size is usually 1 during sampling
        batch_of_sequences = _transform_observation_batch_to_dict_of_batched_sequences(self.observation_space, observations)
        states, actions, returns_to_go, timesteps, attention_mask = (
            batch_of_sequences["states"],
            batch_of_sequences["actions"],
            batch_of_sequences["returns_to_go"],
            batch_of_sequences["timesteps"],
            batch_of_sequences["attention_mask"],
        )

        # Use DT's get_actions (returns [batch_size, act_dim])
        predicted_actions = self.dt_model.get_batch_actions(
            states, actions, returns_to_go, timesteps, attention_mask, batch_size=observations.shape[0], device=self.device
        )
        return predicted_actions
    
    
    def set_training_mode(self, training_mode: bool) -> None:
        self.train(training_mode)
        self.dt_model.train(training_mode)
    

class LastStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, dictionary_structured_observation_space):
        super().__init__(observation_space, features_dim)
        self.dictionary_structured_observation_space = dictionary_structured_observation_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, observations):
        # Extract the last state from the batch of observations
        batch_of_sequences = _transform_observation_batch_to_dict_of_batched_sequences(self.dictionary_structured_observation_space, observations)
        
        return torch.tensor(batch_of_sequences["states"][:, -1, :], dtype=torch.float32, device=self.device)


class DecisionTransformerTD3Policy(TD3Policy):
    def __init__(self, observation_space, action_space, lr_schedule, dt_kwargs, env, **kwargs):
        self.env = env
        self.dt_kwargs = dt_kwargs
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        
    def make_actor(self, features_extractor = None):
        return DecisionTransformerActor(self.dt_kwargs, self.env.env.observation_space)
    
    
    def make_features_extractor(self):
        return LastStateExtractor(self.env.observation_space, self.env.env.env.observation_space.shape[0], self.env.env.observation_space)
