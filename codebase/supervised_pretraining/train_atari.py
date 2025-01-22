import logging

logger = logging.getLogger(__name__)
import argparse
import os
import torch

# Add project's root directory to pythonpath for imports.
import sys
root_directory = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..")
sys.path.append(root_directory)

from components.decision_transformer.atari.mingpt.utils import set_seed
from trainer_atari import FlowTrainer, FlowTrainerConfig
from components.decision_transformer.atari.mingpt.model_atari import GPT, GPTConfig
from components.ale_atari_env.ale_env import ALEModern
from pretrained_agents.atari.get_data import atari_data_generator

# General arguments
parser = argparse.ArgumentParser()

absolute_path = os.path.dirname(__file__)

parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--context_length', type=int, default=30, help="Size of blocks (number of steps in the sequence passed to the transformer).")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--game', type=str, default="Star Gunner")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size - number of blocks.")
parser.add_argument('--ckpt_path', type=str, help="Checkpoint path.")
parser.add_argument('--load_ckpt', type=str, help="Load pretrained checkpoint (specified by path to it) and train it further.")
parser.add_argument('--trainer_model', type=str, help="Model to use to generate train data.", default="model_03750000.gz")

args = parser.parse_args()

# Arguments of dataset generator
dataset_generator_parser = argparse.ArgumentParser()

dataset_generator_parser.add_argument("--batch_size", default=args.batch_size, type=int)
dataset_generator_parser.add_argument("--block_size", default=args.context_length, type=int)

dataset_generator_args = dataset_generator_parser.parse_args([])

# Arguments of pure data generator (atari_data_generator)
pure_data_parser = argparse.ArgumentParser()

game = args.game.replace(" ", "")
absolute_path = os.path.dirname(__file__)
relative_part = os.path.join("pretrained_agents", "atari", "models", "MDQN_modern", game, "0", args.trainer_model)
pure_data_parser.add_argument("--path", default=os.path.join(absolute_path, relative_part), type=str, help="Model path")

pure_data_generator_args = pure_data_parser.parse_args([])

# Set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

logger.info("Starting new experiment: %s", args.game)

if args.seed is not None:
    set_seed(args.seed)

# Maximal length of atari games := 1e4
# (That should be around 2 * maximal length I came across during some tests of pre-trained agents. Of course if the agent will do random things or nothing it might get stuck somewhere and remain there forever, but all three games would eventualy (hopefully) end him, so I suppose this should be enough for at least somehow good agent...)
max_timestep=int(1e4)

logger.info("Creating test environment")

env = ALEModern(
    game,
    torch.randint(100_000, (1,)).item(),
    torch.device('cuda'),
    clip_rewards_val=False,
    sticky_action_p=0,
    sdl=False
)

logger.info("Test environment created: %s", env)

mconf = GPTConfig(env.action_space.n, args.context_length*3,
                  n_layer=6, n_head=8, n_embd=128, model_type='reward_conditioned', max_timestep=max_timestep)
model = GPT(mconf)

# If required, load pretrained agent for further training
if args.load_ckpt is not None:
    logger.info("Loading %s", args.load_ckpt)
    model.load_state_dict(torch.load(args.load_ckpt))

# Initialize a trainer instance and kick off training
# epoch_length = 5e3 with batch_size = 128 should be over 500 thousand updates (64e4)
tconf = FlowTrainerConfig(max_epochs=args.epochs, epoch_length=5e3, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*1000, final_tokens=2*5e5*args.batch_size*args.context_length*3,
                      seed=args.seed, max_timestep=max_timestep, ckpt_path=args.ckpt_path,
                      dataset_generator_arguments=dataset_generator_args,
                      pure_data_generator=atari_data_generator,
                      pure_data_generator_arguments=pure_data_generator_args,
                      test_env=env)
trainer = FlowTrainer(model, tconf)

trainer.train()
