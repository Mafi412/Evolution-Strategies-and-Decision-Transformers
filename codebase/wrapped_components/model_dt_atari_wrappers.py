from es_utilities.wrappers import EsModelWrapper
from es_utilities import optimizers

from components.decision_transformer.atari.mingpt.utils import sample
from components.decision_transformer.atari.mingpt.model_atari import GPT, GPTConfig

import torch


class DTAtari(EsModelWrapper):
    def __init__(self, model, optimizer, state_shape, target_return, sample_action):
        super().__init__(model, optimizer, state_shape, target_return, sample_action)
        
        assert isinstance(sample_action, bool)
        
        self.block_size = model.get_block_size()
        
        assert self.block_size % 3 == 0
        
        self.context_length = self.block_size // 3
        
        self.state_shape = state_shape
        self.action_shape = 1
        self.target_return = target_return
        
        self.state_history_window = torch.zeros((1, 0,) + tuple(self.state_shape), dtype=torch.float32)
        self.action_history_window = None
        self.return_to_go_history_window = torch.zeros((1, 0, 1), dtype=torch.float32)
        self.timesteps = torch.zeros((1, 0, 1), dtype=torch.long)
        
        self.sample_action = sample_action
    
    def choose_action(self, state):
        # Add normalized current state
        state = (state - self.vbn_stats.mean) / self.vbn_stats.std
        state = state.reshape((1,1) + tuple(self.state_shape)).to(dtype=torch.float32)
        self.state_history_window = torch.cat([self.state_history_window, state], dim=1)
        
        if self.state_history_window.size(1) > self.context_length:
            self.state_history_window = self.state_history_window[:,-self.context_length:,:]
            
        action = sample(
            self.model, self.state_history_window, 1,
            sample=self.sample_action,
            actions=self.action_history_window,
            rtgs=self.return_to_go_history_window,
            timesteps=self.timesteps
        )
        action = action.cpu().numpy()[0,-1]
        
        return action
    
    def update_after_step(self, state, next_state, action, reward, terminated, truncated):
        # Update action history window
        action = torch.tensor(action, dtype=torch.long).reshape(1, 1, self.action_shape)
        
        if self.action_history_window is not None:
            self.action_history_window = torch.cat([self.action_history_window, action], dim=1)
        else:
            self.action_history_window = action
            
        if self.action_history_window.size(1) > self.context_length:
            self.action_history_window = self.action_history_window[:,-self.context_length:,:]
            
        # Update return-to-go history window
        new_rtg = (self.return_to_go_history_window[0,-1,0] - reward).reshape(1, 1, 1).to(dtype=torch.float32)
        self.return_to_go_history_window = torch.cat([self.return_to_go_history_window, new_rtg], dim=1)
        
        if self.return_to_go_history_window.size(1) > self.context_length:
            self.return_to_go_history_window = self.return_to_go_history_window[:,-self.context_length:,:]
        
        # Update timesteps
        self.timesteps += 1
    
    def reset_inner_state(self):
        self.state_history_window = torch.zeros((1, 0,) + tuple(self.state_shape), dtype=torch.float32)
        self.action_history_window = None
        self.return_to_go_history_window = torch.tensor(self.target_return, dtype=torch.float32).reshape(1, 1, 1)
        self.timesteps = torch.zeros((1, 1, 1), dtype=torch.long)

    def set_target_return(self, new_target_return):
        self.target_return = new_target_return


def get_new_wrapped_dt(
    state_shape,
    target_return,
    sample_action,
    num_of_actions,
    context_length,
    max_timestep,
    model_initialization_seed=None,
    optimizer_name="ADAM",
    learning_rate=1e-2,
    **kwargs
):
    if model_initialization_seed is not None:
            torch.manual_seed(model_initialization_seed)
            
    mconf = GPTConfig(
        num_of_actions,
        3 * context_length,
        n_layer=6,
        n_head=8,
        n_embd=128,
        model_type='reward_conditioned',
        max_timestep=max_timestep
    )
    model = GPT(mconf)
    
    optimizer = optimizers.create_optimizer_to_model_from_string_name(model, optimizer_name, learning_rate, **kwargs)
    
    return DTAtari(model, optimizer, state_shape, target_return, sample_action)


def get_new_wrapped_dt_for_ale_environment(
    env,
    target_return,
    sample_action,
    context_length,
    max_timestep,
    model_initialization_seed=None,
    optimizer_name="ADAM",
    learning_rate=1e-2,
    **kwargs
):
    return get_new_wrapped_dt(
        (1, env.window, 84, 84),
        target_return,
        sample_action,
        env.action_space.n,
        context_length,
        max_timestep,
        model_initialization_seed,
        optimizer_name,
        learning_rate,
        **kwargs
    )
