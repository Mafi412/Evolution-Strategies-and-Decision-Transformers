import gym

from argparse import ArgumentParser
import os
import math

from wrapped_components.env_gym_mujoco_wrappers import GymMujocoWrapper
from wrapped_components.model_ff_mujoco_wrappers import get_new_wrapped_ff_humanoid
from es_utilities.play import simulate


def main(args):
    env = gym.make("Humanoid-v4", render_mode=("rgb_array" if args.record else (None if args.dont_show_gameplay else "human")))
    
    if args.record is not None:
        if args.record.lower() == "ckpt":
            record_folder = os.path.join(os.path.dirname(args.ckpt_path), "videos")
            video_prefix = "video"
            
        else:
            record_folder = os.path.dirname(args.record)
            video_prefix = os.path.basename(args.record)
            
        env = gym.wrappers.RecordVideo(
            env,
            record_folder,
            name_prefix=video_prefix
        )
    
    wrapped_environment = GymMujocoWrapper(env, args.seed, 1000.)
    
    wrapped_model = get_new_wrapped_ff_humanoid()
    wrapped_model.load_parameters(args.ckpt_path)
            
    episode_returns, episode_lengths = simulate(wrapped_model, wrapped_environment, args.episodes)
    
    if args.save_outputs:
        outputs_folder = os.path.join(os.path.dirname(args.ckpt_path), "performance")
        os.makedirs(outputs_folder, exist_ok=True)
        
        with open(os.path.join(outputs_folder, "performance.csv"), "w") as f:
            f.write("Episode Return;Episode Length\n")
            for episode_return, episode_length in zip(episode_returns, episode_lengths):
                f.write(f"{episode_return};{episode_length}\n")
                
        returns_mean = sum(episode_returns) / len(episode_returns)
        returns_std = math.sqrt(sum((x - returns_mean)**2 for x in episode_returns) / len(episode_returns))
        lengths_mean = sum(episode_lengths) / len(episode_lengths)
        lengths_std = math.sqrt(sum((x - lengths_mean)**2 for x in episode_lengths) / len(episode_lengths))
        
        with open(os.path.join(outputs_folder, f"performance_aggregated.csv"), "w") as f:
            f.write("Mean Return;Std Return;Mean Length;Std Length\n")
            f.write(f"{returns_mean};{returns_std};{lengths_mean};{lengths_std}\n")
    

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=str, help="Checkpoint path.")
    parser.add_argument("-e", "--episodes", default=1, type=int, help="Number of episodes.")
    parser.add_argument("-d", "--dont_show_gameplay", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
    parser.add_argument("-r", "--record", nargs="?", type=str, default=None, const="ckpt", help="Flags whether to record a video. If no value is provided, the recording is saved into a new subfolder \"videos\" of the folder the checkpoint was loaded from. Otherwise, the given value is used as a full path the recordings are saved to.")
    parser.add_argument("-s", "--save_outputs", action="store_true", help="Flags whether to save episode returns and lengths. The data is saved into file \"performance.csv\" in a subfolder \"performance\" of the folder the checkpoint was loaded from, which is created, if necessary. The mean and standard deviation of the data are saved to file \"performance_aggregated.csv\" in the same folder.")
    
    main(parser.parse_args())
