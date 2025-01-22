import gym
import numpy as np
import torch

import argparse
import os

# Add project's root directory to pythonpath for imports.
import sys
root_directory = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..")
sys.path.append(root_directory)

from components.decision_transformer.gym.evaluation.evaluate_episodes import evaluate_episode_rtg
from components.decision_transformer.gym.models.decision_transformer import DecisionTransformer
from pretrained_agents.mujoco.humanoid.get_data import humanoid_data_generator
from pretrained_agents.create_dataset import dataset_generator
from trainer_mujoco import TrainerHumanoid


def training(args):
    device = args.device
    
    env = gym.make("Humanoid-v4")
    max_ep_len = 1000
    scale = 1000. # Normalization for rewards/returns
    env_targets = [10000, 6000, 4000]  # Evaluation conditioning targets

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print('=' * 50)
    print(f'Starting new experiment: Humanoid')

    context_length = args.context_length
    batch_size = args.batch_size
    
    def get_dataset_generator_function():
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", default=batch_size, type=int, help="Batch size - number of blocks.")
        parser.add_argument("--block_size", default=context_length, type=int, help="Size of blocks (number of steps in the sequence passed to the transformer).")
        generator_args = parser.parse_args([])
        
        data_parser = argparse.ArgumentParser()
        data_parser.add_argument("--env", default="Humanoid-v4", type=str, help="Environment.")
        data_parser.add_argument("--seed", default=None, type=int, help="Random seed.")
        data_parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
        data_parser.add_argument("--envs", default=4, type=int, help="Environments.")
        absolute_path = os.path.dirname(__file__)
        relative_part = os.path.join("pretrained_agents", "mujoco", "humanoid", "models", args.trainer_model)
        data_parser.add_argument("--model_path", default=os.path.join(absolute_path, relative_part), type=str, help="Model path")

        pure_data_generator_args = data_parser.parse_args([])
        
        generator = dataset_generator(generator_args, humanoid_data_generator, pure_data_generator_args)
        
        def get_batch(*args, **kwargs):
            batch_states, batch_actions, batch_rtg, batch_timesteps, batch_mask = next(generator)
            
            batch_states = batch_states.to(dtype=torch.float32, device=device)
            batch_actions = batch_actions.to(dtype=torch.float32, device=device)
            batch_rtg = (batch_rtg / scale).to(dtype=torch.float32, device=device)
            batch_timesteps = batch_timesteps.to(dtype=torch.long, device=device)
            batch_mask = batch_mask.to(device=device)
            
            return (batch_states, batch_actions, batch_rtg, batch_timesteps, batch_mask)
        
        return get_batch

    num_eval_episodes = args.num_eval_episodes
    
    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew/scale,
                        mode='normal',
                        state_mean=np.zeros(()),
                        state_std=np.ones(()),
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=context_length,
        max_ep_len=max_ep_len,
        hidden_size=args.embed_dim,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=4*args.embed_dim,
        activation_function=args.activation_function,
        n_positions=1024,
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout,
    )

    model = model.to(device=device)
    
    print(f"Number of parameters of the model: {sum(p.numel() for p in model.parameters())}")
    
    if args.load_ckpt is not None:
        print(f"Loading {args.load_ckpt}")
        model.load_state_dict(torch.load(args.load_ckpt))
    
    print('=' * 50)

    warmup_steps = args.warmup_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    trainer = TrainerHumanoid(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_dataset_generator_function(),
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
        eval_fns=[eval_episodes(tar) for tar in env_targets],
        ckpt_path=args.ckpt_path
    )

    for iter in range(args.max_iters):
        outputs = trainer.train_iteration(num_steps=args.num_steps_per_iter, iter_num=iter+1, print_logs=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context_length', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt_path', type=str, help="Checkpoint path.")
    parser.add_argument('--load_ckpt', type=str, help="Load pretrained checkpoint (specified by path to it) and train it further.")
    parser.add_argument('--trainer_model', type=str, help="Model to use to generate train data.", default="sac_humanoid_4000.model")
    
    args = parser.parse_args()

    training(args=args)
