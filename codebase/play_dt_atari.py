from argparse import ArgumentParser
import random
import os
import math

from components.ale_atari_env.ale_env import ALEModern
from wrapped_components.env_ale_atari_wrappers import ALEAtariWrapper
from wrapped_components.model_dt_atari_wrappers import get_new_wrapped_dt_for_ale_environment
from es_utilities.play import simulate


def main(args):
    main_seed = args.seed
    
    env = ALEModern(
        args.game.replace(" ", ""),
        random.randint(0, 100_000) if main_seed is None else main_seed,
        device="cpu",
        clip_rewards_val=False,
        sticky_action_p=args.sticky_action_p,
        sdl=not args.dont_show_gameplay,
    )
    wrapped_environment = ALEAtariWrapper(env, main_seed)
    
    wrapped_model = get_new_wrapped_dt_for_ale_environment(
        env,
        args.rtg,
        not args.dont_sample_action,
        args.context_length,
        int(1e4),
        main_seed,
        None,
        None
    )
    wrapped_model.train(False)
    wrapped_model.load_parameters(args.ckpt_path)
            
    episode_returns, episode_lengths = simulate(wrapped_model, wrapped_environment, args.episodes)
    
    if args.save_outputs:
        outputs_folder = os.path.join(os.path.dirname(args.ckpt_path), "performance")
        os.makedirs(outputs_folder, exist_ok=True)
        
        with open(os.path.join(outputs_folder, f"{args.rtg}.csv"), "w") as f:
            f.write("Episode Return;Episode Length\n")
            for episode_return, episode_length in zip(episode_returns, episode_lengths):
                f.write(f"{episode_return};{episode_length}\n")
                
        returns_mean = sum(episode_returns) / len(episode_returns)
        returns_std = math.sqrt(sum((x - returns_mean)**2 for x in episode_returns) / len(episode_returns))
        lengths_mean = sum(episode_lengths) / len(episode_lengths)
        lengths_std = math.sqrt(sum((x - lengths_mean)**2 for x in episode_lengths) / len(episode_lengths))
        
        with open(os.path.join(outputs_folder, f"{args.rtg}_aggregated.csv"), "w") as f:
            f.write("Mean Return;Std Return;Mean Length;Std Length\n")
            f.write(f"{returns_mean};{returns_std};{lengths_mean};{lengths_std}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=str, help="Checkpoint path.")
    parser.add_argument("rtg", type=int, help="Return-to-go that should be passed.")
    parser.add_argument("--context_length", type=int, default=30, help="Size of blocks (number of steps in the sequence passed to the transformer).")
    parser.add_argument("--game", type=str, default="Hero")
    parser.add_argument("-e", "--episodes", default=1, type=int, help="Number of episodes.")
    parser.add_argument("-d", "--dont_show_gameplay", action="store_true")
    parser.add_argument("--dont_sample_action", action="store_true")
    parser.add_argument("--sticky_action_p", type=float, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
    parser.add_argument("-s", "--save_outputs", action="store_true", help="Flags whether to save episode returns and lengths. The data is saved into file \"{rtg}.csv\" in a subfolder \"performance\" of the folder the checkpoint was loaded from, which is created, if necessary. The mean and standard deviation of the data are saved to file \"{rtg}_aggregated.csv\" in the same folder.")
    
    main(parser.parse_args())
