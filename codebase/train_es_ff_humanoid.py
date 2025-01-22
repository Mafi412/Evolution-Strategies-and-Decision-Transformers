from es.es import es
from wrapped_components.model_ff_mujoco_wrappers import get_new_wrapped_ff_humanoid
from wrapped_components.env_gym_mujoco_wrappers import GymMujocoWrapper

import gym

import os
from argparse import ArgumentParser


def main(args):
    # Gym does sometimes output those, but we don't really care, they just clog our output.
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    main_seed = args.seed
    env = gym.make("Humanoid-v4", render_mode=None)
    test_environment = GymMujocoWrapper(env, main_seed, 1000.)
    model = get_new_wrapped_ff_humanoid(main_seed, args.optimizer, args.learning_rate)
    size_of_population = args.size_of_population
    num_of_iterations = args.num_of_iterations
    noise_deviation = args.noise_deviation
    weight_decay_factor = args.weight_decay_factor
    batch_size = args.batch_size
    update_vbn_stats_probability = args.update_vbn_stats_probability
    path_for_checkpoints = os.path.join(args.path_to_folder_for_logs_and_checkpoints, "ckpts", "ckpt")
    logging_path = os.path.join(args.path_to_folder_for_logs_and_checkpoints, "log")
    
    resulting_model = es(
        model,
        test_environment,
        size_of_population,
        num_of_iterations,
        main_seed,
        noise_deviation,
        weight_decay_factor,
        batch_size,
        update_vbn_stats_probability,
        path_for_checkpoints,
        logging_path
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path_to_folder_for_logs_and_checkpoints", type=str, help="Path to the folder, which the checkpoint and log files will be stored into. (It will be created, if it does not exist.)")
    parser.add_argument("--size_of_population", type=int, default=5000, help="Size of population, or more precisely number of tested noises within one generation / iteration.")
    parser.add_argument("--num_of_iterations", type=int, default=200, help="Number of iterations the ES will run.")
    parser.add_argument("--seed", type=int, default=None, help="Main seed.")
    parser.add_argument("--noise_deviation", type=float, default=0.02, help="Deviation of the noise added during training.")
    parser.add_argument("--weight_decay_factor", type=float, default=0.995, help="Factor of the weight decay.")
    parser.add_argument("--batch_size", type=int, default=1000, help="A size of a batch for a batched weighted sum of noises during model update.")
    parser.add_argument("--update_vbn_stats_probability", type=float, default=0.01, help="How often to use data obtained during evaluation to update the Virtual Batch Norm stats.")
    parser.add_argument("--optimizer", type=str, default="SGDM", help="Optimizer to be used. Either \"ADAM\", or \"SGDM\" (standing for SGD with Momentum), or \"SGD\".")
    parser.add_argument("--learning_rate", type=float, default=5e-2, help="Learning rate (or could be called step size).")
    
    main(parser.parse_args())
