from es_utilities.wrappers import EsModelWrapper
from es_utilities import optimizers

from components.decision_transformer.gym.models.decision_transformer import DecisionTransformer

import torch


class DTMujoco(EsModelWrapper):
    def __init__(self, model, optimizer, state_shape, action_shape, target_return):
        super().__init__(model, optimizer, state_shape, action_shape, target_return)
        
        state_dimension = state_shape[0]
        action_dimension = action_shape[0]
        assert state_dimension == model.state_dim
        assert action_dimension == model.act_dim
        
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        self.target_return = target_return
        
        self.state_history = torch.zeros((0, self.state_dimension), dtype=torch.float32)
        self.action_history = torch.zeros((0, self.action_dimension), dtype=torch.float32)
        self.return_to_go_history = torch.zeros((0,), dtype=torch.float32)
        self.timesteps_history = torch.zeros((0,), dtype=torch.long)
    
    def choose_action(self, state):
        # Add normalized current state (We keep whole history - unlike in the Atari, here it does not slow down the program)
        state = (state - self.vbn_stats.mean) / self.vbn_stats.std
        current_state = torch.from_numpy(state).reshape(1, self.state_dimension).to(dtype=torch.float32)
        self.state_history = torch.cat([self.state_history, current_state], dim=0)

        # Add action padding
        self.action_history = torch.cat([self.action_history, torch.zeros((1, self.action_dimension), dtype=torch.float32)], dim=0)
        
        action = self.model.get_action(
            self.state_history.to(dtype=torch.float32),
            self.action_history.to(dtype=torch.float32),
            None,
            self.return_to_go_history.to(dtype=torch.float32),
            self.timesteps_history.to(dtype=torch.long),
        )
        action = action.detach().cpu().numpy()
        
        return action
    
    def update_after_step(self, state, next_state, action, reward, terminated, truncated):
        # Update action history
        self.action_history[-1] = torch.from_numpy(action).to(dtype=torch.float32)
        
        # Update return-to-go history
        predicted_return = self.return_to_go_history[-1] - reward
        self.return_to_go_history = torch.cat([self.return_to_go_history, predicted_return.reshape(1,).to(dtype=torch.float32)], dim=0)
        
        # Update timesteps history
        self.timesteps_history = torch.cat(
            [self.timesteps_history, torch.ones((1,), dtype=torch.long) * (self.timesteps_history[-1]+1)], dim=0
        )
    
    def reset_inner_state(self):
        self.state_history = torch.zeros((0, self.state_dimension), dtype=torch.float32)
        self.action_history = torch.zeros((0, self.action_dimension), dtype=torch.float32)
        self.return_to_go_history = torch.tensor([self.target_return], dtype=torch.float32)
        self.timesteps_history = torch.zeros((1,), dtype=torch.long)

    def set_target_return(self, new_target_return):
        self.target_return = new_target_return


class DTMujocoFixedLN(DTMujoco):
    def __init__(self, model, optimizer, state_shape, action_shape, target_return):
        super().__init__(model, optimizer, state_shape, action_shape, target_return)
        
    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
        for name, param in super().named_parameters(prefix, recurse, remove_duplicate):
            if "ln_" in name or "_ln" in name:
                continue
            yield name, param


def get_new_wrapped_dt(
    state_dimension,
    action_dimension,
    target_return,
    timestep_limit=1000,
    context_length=20,
    embed_dim=128,
    n_layer=3,
    n_head=1,
    activation_function="relu",
    dropout=0.1,
    fixed_layer_norm=False,
    model_initialization_seed=None,
    optimizer_name="ADAM",
    learning_rate=1e-2,
    **kwargs
):
    if model_initialization_seed is not None:
            torch.manual_seed(model_initialization_seed)
            
    model = DecisionTransformer(
        state_dim=state_dimension,
        act_dim=action_dimension,
        max_length=context_length,
        max_ep_len=timestep_limit,
        hidden_size=embed_dim,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4*embed_dim,
        activation_function=activation_function,
        n_positions=1024,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
    )
    
    optimizer = optimizers.create_optimizer_to_model_from_string_name(model, optimizer_name, learning_rate, **kwargs)
    
    if not fixed_layer_norm:
        return DTMujoco(model, optimizer, (state_dimension,), (action_dimension,), target_return)
    
    else:
        return DTMujocoFixedLN(model, optimizer, (state_dimension,), (action_dimension,), target_return)


def get_new_wrapped_dt_humanoid(
    target_return,
    timestep_limit=1000,
    context_length=20,
    embed_dim=128,
    n_layer=3,
    n_head=1,
    activation_function="relu",
    dropout=0.1,
    fixed_layer_norm=False,
    model_initialization_seed=None,
    optimizer_name="ADAM",
    learning_rate=1e-2,
    **kwargs
):
    return get_new_wrapped_dt(
        376,
        17,
        target_return,
        timestep_limit,
        context_length,
        embed_dim,
        n_layer,
        n_head,
        activation_function,
        dropout,
        fixed_layer_norm,
        model_initialization_seed,
        optimizer_name,
        learning_rate,
        **kwargs
    )
