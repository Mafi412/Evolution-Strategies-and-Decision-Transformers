import torch

from components.decision_transformer.gym.training.trainer import Trainer


class TrainerHumanoid(Trainer):
    
    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None, ckpt_path=None):
        super().__init__(model, optimizer, batch_size, get_batch, loss_fn, scheduler, eval_fns)
        
        self.ckpt_path = ckpt_path
    
    
    def train_iteration(self, num_steps, iter_num=0, print_logs=False):
        # Training
        print(f"Training iteration {iter_num}")
        logs = super().train_iteration(num_steps, iter_num, print_logs)
            
        # Saving the model
        path = self.ckpt_path + "_" + str(iter_num) + ".ckpt"
        print(f"Saving to {path}")
        torch.save(self.model.state_dict(), path)
        
        return logs
    

    def train_step(self):
        states, actions, rtg, timesteps, attention_mask = self.get_batch()
        action_target = torch.clone(actions)
        rewards = None

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
