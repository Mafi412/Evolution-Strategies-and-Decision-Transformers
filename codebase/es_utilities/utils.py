# Utilities, like shared noise table, functions for working with noise, etc.

import random

import numpy as np
import torch


# --- Shared noise table ---

class SharedNoiseTable:
    def __init__(self, deviation=1, seed=None):
        self._length = int(25e7)  # 1 gigabyte of 32-bit numbers.
        self.noise = np.empty(self._length, dtype="f")
        self.noise[:] = np.random.default_rng(seed).normal(0, deviation, self._length)

    def get(self, i, dim):
        if i + dim <= self._length:
            return self.noise[i:i + dim]
        else:
            return np.concatenate((self.noise[i:], self.noise[:(i + dim - self._length)]))
    
    def __len__(self):
        return self._length
    
    
def get_noise(model, noise_table, seed=None):
    if seed is not None:
        random.seed(seed)
        
    noise = dict()
    for name, param in model.named_parameters():
        number_of_elements, shape = param.numel(), param.size()
        current_noise = noise_table.get(random.randint(0, len(noise_table) - 1), number_of_elements)
        noise[name] = torch.from_numpy(np.reshape(current_noise, shape))
        
    return noise


def add_noise(model, noise):
    for name, param in model.named_parameters():
        param.add_(noise[name])
    

def subtract_noise(model, noise):
    add_noise(model, get_minus_noise(noise))
    
    
def get_minus_noise(noise):
    minus_noise = dict()
    for key in noise:
        minus_noise[key] = -noise[key]
        
    return minus_noise

    
def get_weighted_noise(weight, noise):
    weighted_noise = dict()
    for key in noise:
        weighted_noise[key] = weight * noise[key]
        
    return weighted_noise


def get_combined_noises(noises):
    combined_noise = dict()
    for key in noises[0]:
        combined_noise[key] = torch.sum(torch.stack([noises[i][key] for i in range(len(noises))]), axis=0)
        
    return combined_noise


# --- General utilities ---        
        
def decay_weights(weight_decay_factor, model):
    for param in model.parameters():
        param.multiply_(weight_decay_factor)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
