from argparse import ArgumentParser

import torch

# Add project's root directory to pythonpath for imports.
import os
import sys
root_directory = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..")
sys.path.append(root_directory)

from components.decision_transformer.atari.mingpt.model_atari import GPT, GPTConfig
from components.decision_transformer.atari.mingpt.utils import sample

from components.ale_atari_env.ale_env import ALEModern


def main(args):
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    
    # Set env
    env = ALEModern(
        args.game.replace(" ", ""),
        torch.randint(100_000, (1,)).item(),
        device=device,
        clip_rewards_val=False,
        sticky_action_p=args.sticky_action_p,
        sdl=not args.dont_show_gameplay,
    )

    # Sanity check
    print(env)

    # Create model:    
    # Maximal length of atari games := 1e4
    # (That should be around 2 * maximal length I came across during some tests of pre-trained agents. Of course if the agent will do random things or nothing it might get stuck somewhere and remain there forever, but all three games would eventualy (hopefully) end him, so I suppose this should be enough for at least somehow good agent...)
    max_timestep=int(1e4)

    mconf = GPTConfig(env.action_space.n, args.context_length*3,
                      n_layer=6, n_head=8, n_embd=128, model_type='reward_conditioned', max_timestep=max_timestep)
    model = GPT(mconf)
    model = model.to(device)

    # Load model
    model.load_state_dict(torch.load(args.ckpt_path))
    model.train(False)

    ep_returns = []

    for ep in range(args.episodes):
        state, reward_sum, done = env.reset(), 0, False
        state = state.type(torch.float32).to(device)
        rtgs = [args.rtg]
        # First state is from env, first rtg is target return, and first timestep is 0
        sampled_action = sample(model, state.unsqueeze(0), 1, temperature=1.0, sample=(not args.dont_sample_action), actions=None, 
            rtgs=torch.tensor(rtgs, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(-1), 
            timesteps=torch.zeros((1, 1, 1), dtype=torch.int64, device=device))

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
                ep_returns.append(reward_sum)
                break
                    
            state = state.to(device)

            if all_states.shape[0] < args.context_length:
                all_states = torch.cat([all_states, state], dim=0)
            else:
                all_states = torch.cat([all_states[-args.context_length+1:], state], dim=0)
                
            if len(actions) > args.context_length:
                actions = actions[-args.context_length:]

            rtgs += [rtgs[-1] - reward]
            if len(rtgs) > args.context_length:
                rtgs = rtgs[-args.context_length:]
            
            # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
            # timestep is just current timestep
            sampled_action = sample(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=(not args.dont_sample_action), 
                actions=torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(-1), 
                rtgs=torch.tensor(rtgs, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(-1), 
                timesteps=(min(j, max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64, device=device)))
            
        print(f"Episode {ep:02d})  Return: {ep_returns[ep]:7.1f} (Timesteps: {j})")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('ckpt_path', type=str, help="Checkpoint path.")
    parser.add_argument('rtg', type=int, help="Return-to-go that should be passed.")
    parser.add_argument('--context_length', type=int, default=30, help="Size of blocks (number of steps in the sequence passed to the transformer).")
    parser.add_argument('--game', type=str, default="Star Gunner")
    parser.add_argument("-e", "--episodes", default=1, type=int, help="Number of episodes.")
    parser.add_argument("--dont_show_gameplay", action="store_true")
    parser.add_argument("--dont_sample_action", action="store_true")
    parser.add_argument('--sticky_action_p', type=float, default=0)
    
    main(parser.parse_args())
