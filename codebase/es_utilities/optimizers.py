# Optimizers

import math
import torch

from . import utils


class Optimizer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate

    def update(self, noise_to_add):
        noise_to_add_with_adjusted_keys = dict(((key[6:], value) for (key, value) in noise_to_add.items())) # Getting rid of the "model." part of the parameter name
        final_noise = self.compute_step(noise_to_add_with_adjusted_keys)
        utils.add_noise(self.model, final_noise)

    def compute_step(self, noise_to_add):
        raise NotImplementedError
    
    def copy_with_new_model(self, model):
        new_optimizer = object.__new__(type(self))
        for key, value in self.__dict__.items():
            if key != "model":
                setattr(new_optimizer, key, value)
            else:
                setattr(new_optimizer, "model", model)
                
        return new_optimizer


class SGD(Optimizer):
    def __init__(self, model, learning_rate):
        Optimizer.__init__(self, model, learning_rate)

    def compute_step(self, noise_to_add):
        step = utils.get_weighted_noise(self.learning_rate, noise_to_add)
        return step


class SGDMomentum(Optimizer):
    def __init__(self, model, learning_rate, momentum=0.9):
        Optimizer.__init__(self, model, learning_rate)
        self.memory = {name: torch.zeros(param.size(), dtype=torch.float32) for name, param in model.named_parameters()}
        self.momentum = momentum

    def compute_step(self, noise_to_add):
        for name in noise_to_add:
            self.memory[name] = self.momentum * self.memory[name] + (1 - self.momentum) * noise_to_add[name]
            
        step = utils.get_weighted_noise(self.learning_rate, self.memory)
        return step


class Adam(Optimizer):
    def __init__(self, model, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        Optimizer.__init__(self, model, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.memory = {name: torch.zeros(param.size(), dtype=torch.float32) for name, param in model.named_parameters()}
        self.memory_squares = {name: torch.zeros(param.size(), dtype=torch.float32) for name, param in model.named_parameters()}

    def compute_step(self, noise_to_add):
        self.t += 1
        bias_correction_constant = math.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        step = dict()
        for name in noise_to_add:
            self.memory[name] = self.beta1 * self.memory[name] + (1 - self.beta1) * noise_to_add[name]
            self.memory_squares[name] = self.beta2 * self.memory_squares[name] + (1 - self.beta2) * (noise_to_add[name] * noise_to_add[name])
            step[name] = bias_correction_constant * self.memory[name] / (torch.sqrt(self.memory_squares[name]) + self.epsilon)
            
        step = utils.get_weighted_noise(self.learning_rate, step)
        return step
    
    
def create_optimizer_to_model_from_string_name(model, optimizer_name, learning_rate, **kwargs):
    if optimizer_name == "ADAM":
        optimizer = Adam(model, learning_rate, **dict(((parameter, kwargs[parameter]) for parameter in ["beta1", "beta2", "epsilon"] if parameter in kwargs)))
        
    elif optimizer_name == "SGDM":
        optimizer = SGDMomentum(model, learning_rate, **dict(((parameter, kwargs[parameter]) for parameter in ["momentum"] if parameter in kwargs)))
            
    elif optimizer_name == "SGD":
        optimizer = SGD(model, learning_rate)
        
    elif optimizer_name is None:
        optimizer = None
        
    else:
        raise ValueError(optimizer_name + " is not a valid optimizer name in this case. (Only ADAM, SGDM (standing for SGD with momentum) and SGD are allowed.)")
    
    return optimizer
