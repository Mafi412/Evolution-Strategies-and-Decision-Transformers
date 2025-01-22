import gym
import torch

from argparse import ArgumentParser
import os

# Add project's root directory to pythonpath for imports.
import sys
root_directory = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..")
sys.path.append(root_directory)

from components.decision_transformer.gym.models.decision_transformer import DecisionTransformer


def main(args):
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        
    env = gym.make("Humanoid-v4", render_mode=(None if args.dont_show_gameplay else "human"))
    max_ep_len = 1000
    scale = 1000. # Normalization for rewards/returns
    
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=args.context_length,
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
    model.load_state_dict(torch.load(args.ckpt_path))
    model.train(False)
    
    ep_returns = []

    for ep in range(args.episodes):
        state, _ = env.reset()

        # We keep all the histories on the device - unlike in the Atari, here it does not slow down the program
        # Note that the latest action will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        
        target_return = torch.tensor((args.rtg / scale), device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    
        episode_return = 0
        for t in range(max_ep_len):

            # Add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)

            action = model.get_action(
                states.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                None,
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)

            pred_return = target_return[0,-1] - (reward/scale)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)],
                dim=1
            )

            episode_return += reward

            if done:
                ep_returns.append(episode_return)
                print(f"Episode {ep:02d})  Return: {ep_returns[ep]:7.1f} (Timesteps: {t+1})")
                break
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('ckpt_path', type=str, help="Checkpoint path.")
    parser.add_argument('rtg', type=float, help="Return-to-go that should be passed.")
    parser.add_argument('--context_length', type=int, default=20, help="Size of blocks (number of steps in the sequence passed to the transformer).")
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("-e", "--episodes", default=1, type=int, help="Number of episodes.")
    parser.add_argument("--dont_show_gameplay", action="store_true")
    
    main(parser.parse_args())
    # The ignored exception is not my fault and even though I tried, I cannot get rid of it... 
    # (Only thing that helps is writing directly into mujoco_rendering.py, which is a library file.)