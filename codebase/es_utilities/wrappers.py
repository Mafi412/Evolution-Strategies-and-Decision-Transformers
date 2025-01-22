# Model and environment wrappers abstract classes

import copy
import os

import numpy as np
import torch
        
        
# --- Stats holder class for virtual batch normalization ---

class VirtualBatchNormalizationStats:
    def __init__(self, shape, eps=1e-2):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sum_of_squares = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, sum, sum_of_squares, count):
        self.sum += sum
        self.sum_of_squares += sum_of_squares
        self.count += count

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sum_of_squares / self.count - np.square(self.mean), 1e-2))
    
    @property
    def shape(self):
        return self.sum.shape

    def set(self, mean, std, count):
        self.sum[:] = mean * count
        self.sum_of_squares[:] = (np.square(mean) + np.square(std)) * count
        self.count = count
        

# --- "Abstract" model wrapper ---

class EsModelWrapper(torch.nn.Module):
    def __init__(self, model, optimizer, state_shape, *args, **kwargs):
        super().__init__()
        if optimizer is not None:
            assert model is optimizer.model
        self.model = model
        self.optimizer = optimizer
        self.vbn_stats = VirtualBatchNormalizationStats(state_shape)
        self.args = args
        self.kwargs = kwargs
        
        # Setting requires_grad of every layer of the model to False (needed for inplace operations on weight tensors of the model)
        for param in self.model.parameters():
            param.requires_grad_(False)
        
    def forward(self, *args, **kwargs):
        self.model.forward(*args, **kwargs)
        
    def clone(self):
        model = copy.deepcopy(self.model)
        
        if self.optimizer is not None:
            optimizer = self.optimizer.copy_with_new_model(model)
        else:
            optimizer = None
        
        result = type(self)(model, optimizer, self.vbn_stats.shape, *self.args, **self.kwargs)
        result.vbn_stats.set(self.vbn_stats.mean, self.vbn_stats.std, self.vbn_stats.count)
        
        return result
    
    def copy_from(self, reference_model):
        self.load_state_dict(reference_model.state_dict())
        self.vbn_stats.set(reference_model.vbn_stats.mean, reference_model.vbn_stats.std, reference_model.vbn_stats.count)
    
    def choose_action(self, state): # Return action to be taken depending on the current state (obtained) and possibly some inner variables or memory
        raise NotImplementedError()
    
    def update_after_step(self, state, next_state, action, reward, terminated, truncated): # Update the inner state
        raise NotImplementedError()
    
    def reset_inner_state(self): # Reset the inner state (before starting new episode)
        raise NotImplementedError()
    
    def save_parameters(self, path, suffix=None):
        path = path + ("" if suffix is None else ("_" + str(suffix)))
        model_path = path + ".model"
        vbn_stats_path = path + ".vbns"
        
        # Save model weights themselves
        torch.save(self.state_dict(), model_path)
        
        # Save VBN stats
        with open(vbn_stats_path, "wb") as f:
            np.savez(f, mean=self.vbn_stats.mean, std=self.vbn_stats.std, count=self.vbn_stats.count)
    
    def load_parameters(self, path):
        model_path = path + ".model"
        vbn_stats_path = path + ".vbns"
        
        if os.path.isfile(model_path):
            # Load model weights themselves
            self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

            # Load VBN stats
            if os.path.isfile(vbn_stats_path):
                with np.load(vbn_stats_path) as data:
                    mean = data["mean"]
                    std = data["std"]
                    count = data["count"].item()
                self.vbn_stats.set(mean, std, count)
                
        else:
            assert os.path.isfile(path)
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        

# --- "Abstract" environment wrapper ---

class EsEnvironmentWrapper:
    def __init__(self, env, seed=None):
        self.env = env
        self.set_seed(seed)
        
    def reset(self): # Reset the environment and return just the new state
        raise NotImplementedError()
    
    def step(self, action):  # Perform one step of the environment and return next_state, reward, terminated, truncated
        raise NotImplementedError()
    
    def set_seed(self, seed):
        raise NotImplementedError()
    
    @property
    def state_shape(self):
        return self.env.observation_space.shape
    
    @property
    def timestep_limit(self): # Returns maximal timestep for the environment (or None if unrestricted)
        raise NotImplementedError()
        