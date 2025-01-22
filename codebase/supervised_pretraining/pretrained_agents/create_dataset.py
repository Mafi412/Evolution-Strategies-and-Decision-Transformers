#!/usr/bin/env python3

from argparse import ArgumentParser

from .atari.get_data import atari_data_generator
from .mujoco.humanoid.get_data import humanoid_data_generator

import random
import numpy as np
import torch

parser = ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size - number of blocks.")
parser.add_argument("--block_size", default=30, type=int, help="Size of blocks (number of steps in the sequence passed to the transformer).")


def dataset_generator(args, pure_data_generator, pure_data_generator_args):
    buffered_data_blocks = list()
    
    for episode in pure_data_generator(pure_data_generator_args):
        
        # create returns_to_go from rewards
        return_to_go = 0
        returns_to_go = list()
        for step in range(len(episode)-1, -1, -1):
            transition = episode[step]
            return_to_go += transition.reward
            returns_to_go.append(return_to_go)
        
        ## padding short episodes by zeros (in the end from the left - for the compliance with the original implementation of continuous MuJoCo decision transformer)
        if len(episode) < args.block_size:
            for _ in range(args.block_size - len(episode)):
                returns_to_go.append(0)
            
        returns_to_go.reverse()
        
        
        # create state and action lists
        states, actions, mask = list(), list(), list()
        
        ## padding short episodes by zeros (from the left - for the compliance with the original implementation of continuous MuJoCo decision transformer)
        if len(episode) < args.block_size:
            for _ in range(args.block_size - len(episode)):
                states.append(np.zeros_like(episode[0].state))
                actions.append(np.zeros_like(episode[0].action))
                mask.append(np.zeros(()))
        
        for step in episode:
            states.append(np.array(step.state))
            actions.append(np.array(step.action))
            mask.append(np.ones(()))
        
        
        # create timestep list
        if len(episode) >= args.block_size:
            timesteps = np.arange(len(episode), dtype=np.int64)
        else:
            timesteps = np.concatenate(
                np.arange(len(episode), dtype=np.int64),
                np.zeros((args.block_size - len(episode),), dtype=np.int64)
            )
        
        # buffer the context blocks from processed episode
        for i in range(len(states) - args.block_size + 1):
            buffered_data_blocks.append((
                states[i:i+args.block_size],
                actions[i:i+args.block_size],
                returns_to_go[i:i+args.block_size],
                (timesteps[i:i+1] if pure_data_generator is atari_data_generator else timesteps[i:i+args.block_size]),
                mask[i:i+args.block_size]
            ))
            
        # shuffle the buffered data
        random.shuffle(buffered_data_blocks)
        
        # creating batch, if we already have enough data
        while len(buffered_data_blocks) >= args.batch_size:
            batch_states = list()
            batch_actions = list()
            batch_rtg = list()
            batch_timesteps = list()
            batch_mask = list()
            for state_block, action_block, returns_to_go_block, timestep_block, mask_block in buffered_data_blocks:
                batch_states.append(state_block)
                batch_actions.append(action_block)
                batch_rtg.append(returns_to_go_block)
                batch_timesteps.append(timestep_block)
                batch_mask.append(mask_block)
                
                if len(batch_states) >= args.batch_size:
                    break
            
            buffered_data_blocks = buffered_data_blocks[args.batch_size:]
            
            batch_states = torch.from_numpy(np.array(batch_states))
            batch_actions = torch.from_numpy(np.array(batch_actions))
            batch_rtg = torch.from_numpy(np.array(batch_rtg)).unsqueeze(-1)
            batch_timesteps = torch.from_numpy(np.array(batch_timesteps))
            batch_mask = torch.from_numpy(np.array(batch_mask))
            
            if pure_data_generator is atari_data_generator: # unsqueeze only if we have atari generator -> action and timesteps is scalar (rtgs are always scalars) (atari -> We need int64.)
                batch_actions = batch_actions.unsqueeze(-1).type(torch.int64)
                batch_timesteps = batch_timesteps.unsqueeze(-1)
            
            yield (batch_states, batch_actions, batch_rtg, batch_timesteps, batch_mask)
            