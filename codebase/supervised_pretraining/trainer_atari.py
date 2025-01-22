"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.

Differs from the original atari trainer mainly by the fact it uses our dataset_generator from create_dataset.py, hence not having epochs (or having them defined by their length) and obtaining new and new data on the fly. (And is edited such that it uses actual instruments / packages instead of deprecated ones.)
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch

logger = logging.getLogger(__name__)

from components.decision_transformer.atari.mingpt.utils import sample

from pretrained_agents.create_dataset import dataset_generator
from pretrained_agents.atari.get_data import atari_data_generator

class FlowTrainerConfig:
    # Optimization parameters
    max_epochs = 10
    epoch_length = 1e4 # Number of iterations per epoch
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # Learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = True
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # Checkpoint settings
    ckpt_path = None
    # Data generation parameters
    dataset_generator_arguments = None
    pure_data_generator = atari_data_generator
    pure_data_generator_arguments = None
    # Environment for the model to be tested in
    test_env = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class FlowTrainer:

    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)


    def save_checkpoint(self, suffix=None):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        path = self.config.ckpt_path + (("_" + suffix) if suffix is not None else "") + ".ckpt"
        logger.info("Saving %s", path)
        torch.save(raw_model.state_dict(), path)


    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            logger.info("Running epoch %s", epoch_num+1)
            
            is_train = split == 'train'
            model.train(is_train)
            data = dataset_generator(
                config.dataset_generator_arguments,
                config.pure_data_generator,
                config.pure_data_generator_arguments
            )

            losses = []
            pbar = tqdm(enumerate(data, start=1), total=config.epoch_length) if is_train else enumerate(data)
            for it, (x, y, r, t, _) in pbar:
                
                # End the epoch when maximum number of iterations is reached
                if it > config.epoch_length:
                    break

                # Place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # Forward the model
                with torch.set_grad_enabled(is_train):
                    _, loss = model(x, y, y, r, t)
                    loss = loss.mean() # Collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # Backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # Decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += torch.numel(y)
                        if self.tokens < config.warmup_tokens:
                            # Linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # Cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # Report progress
                    pbar.set_description(f"Epoch {epoch+1}, iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                    
                    # Store best yet encountered return-to-go
                    actual_best_return_to_go = torch.max(r).item()
                    if self.best_return_to_go_in_data < actual_best_return_to_go:
                        self.best_return_to_go_in_data = actual_best_return_to_go

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("Test loss: %f", test_loss)
                return test_loss

        self.tokens = 0 # Counter used for learning rate decay
        self.best_return_to_go_in_data = -float('inf') # Best return-to-go yet in the generated data, which will be passed to evaluation

        for epoch in range(config.max_epochs):

            run_epoch('train', epoch_num=epoch)
            
            self.save_checkpoint(str(epoch+1))
            
            self.get_returns(self.best_return_to_go_in_data)


    def get_returns(self, ret):
        logger.info("Validating the model")
        
        self.model.train(False)
        env = self.config.test_env
        num_of_episodes = 10

        T_rewards = []
        for _ in range(num_of_episodes):
            state, reward_sum, done = env.reset(), 0, False
            state = state.type(torch.float32).to(self.device)
            rtgs = [ret]
            # First state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(self.model.module, state.unsqueeze(0), 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64, device=self.device))

            j = 0
            all_states = state
            actions = []
            while True:
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.to(self.device)

                if all_states.shape[0] < self.config.dataset_generator_arguments.block_size:
                    all_states = torch.cat([all_states, state], dim=0)
                else:
                    all_states = torch.cat([all_states[-self.config.dataset_generator_arguments.block_size+1:], state], dim=0)

                if len(actions) > self.config.dataset_generator_arguments.block_size:
                    actions = actions[-self.config.dataset_generator_arguments.block_size:]

                rtgs += [rtgs[-1] - reward]
                if len(rtgs) > self.config.dataset_generator_arguments.block_size:
                    rtgs = rtgs[-self.config.dataset_generator_arguments.block_size:]
                
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(-1), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64, device=self.device)))
        
        env.close()
        eval_return = sum(T_rewards)/float(num_of_episodes)
        logger.info("Target return: %d, Eval return: %d", ret, eval_return)
        self.model.train(True)
        return eval_return
