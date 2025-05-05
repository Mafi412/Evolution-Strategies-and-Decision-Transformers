from argparse import ArgumentParser
from pathlib import Path

from td3_dt.dt_for_td3 import DecisionTransformerTD3Policy
from td3_dt.dt_humanoid_td3_env import get_flattened_dt_humanoid_env

from stable_baselines3 import TD3


def main(args):
    env = get_flattened_dt_humanoid_env(args.context_length, args.rtg)
    
    dt_kwargs = dict(
        state_dim=376,
        act_dim=17,
        max_length=args.context_length,
        max_ep_len=1000,
        hidden_size=args.embed_dim,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=4*args.embed_dim,
        activation_function=args.activation_function,
        n_positions=1024,
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout,
    )

    policy_kwargs = dict(
        dt_kwargs=dt_kwargs,
        env=env,
    )

    model = TD3(
        policy=DecisionTransformerTD3Policy,
        env=env,
        policy_kwargs=policy_kwargs,
        buffer_size=args.replay_buffer_size,
        seed=args.seed,
        verbose=1,
    )

    model.learn(
        total_timesteps=1_000_000,
        log_interval=args.log_interval
    )
    model.save(Path(args.path_to_folder_to_save_the_model) / "td3_dt_humanoid.td3_model")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path_to_folder_to_save_the_model", type=str, help="Path to the folder, which the checkpoint and log files will be stored into. (It will be created, if it does not exist.)")
    parser.add_argument("rtg", type=float, help="Return-to-go that should be passed.")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging.")
    parser.add_argument("--seed", type=int, default=None, help="Main seed.")
    parser.add_argument("--replay_buffer_size", type=int, default=1000000, help="Size of the replay buffer.")
    parser.add_argument("--context_length", type=int, default=20, help="Size of blocks (number of steps in the sequence passed to the transformer).")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    
    main(parser.parse_args())
