from es_utilities.wrappers import EsModelWrapper
from es_utilities import optimizers

import numpy as np
import torch


class FFMujoco(EsModelWrapper):
    def __init__(self, model, optimizer, state_shape):
        super().__init__(model, optimizer, state_shape)
        
    def choose_action(self, state):
        normalized_state = (state - self.vbn_stats.mean) / self.vbn_stats.std
        action = self.model(torch.from_numpy(normalized_state)).numpy()
        return action
    
    def update_after_step(self, state, next_state, action, reward, terminated, truncated):
        pass
    
    def reset_inner_state(self):
        pass
    
    
class FFModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, inner_dim, model_initialization_seed=None):
        super().__init__()
        
        if model_initialization_seed is not None:
            torch.manual_seed(model_initialization_seed)
        
        self.hidden_layer1 = torch.nn.Linear(input_dim, inner_dim)
        self.hidden_layer2 = torch.nn.Linear(inner_dim, inner_dim)
        self.output_layer = torch.nn.Linear(inner_dim, output_dim)
        
        self.activation = torch.tanh
        
        self = self.double()
        
    def forward(self, x):
        x = self.activation(self.hidden_layer1(x))
        x = self.activation(self.hidden_layer2(x))
        return self.activation(self.output_layer(x))
    

def get_new_wrapped_ff(input_dim, output_dim, inner_dim, model_initialization_seed=None, optimizer_name="ADAM", learning_rate=1e-2, **kwargs):
    model = FFModel(input_dim, output_dim, inner_dim, model_initialization_seed)
    
    optimizer = optimizers.create_optimizer_to_model_from_string_name(model, optimizer_name, learning_rate, **kwargs)
    
    return FFMujoco(model, optimizer, (input_dim,))
    
    
def get_new_wrapped_ff_humanoid(model_initialization_seed=None, optimizer_name="ADAM", learning_rate=1e-2, **kwargs):
    return get_new_wrapped_ff(376, 17, 256, model_initialization_seed, optimizer_name, learning_rate, **kwargs)
