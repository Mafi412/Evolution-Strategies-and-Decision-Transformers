import collections
from functools import partial
from gzip import GzipFile
from pathlib import Path

import torch
from torch import nn

from components.ale_atari_env.ale_env import ALEModern, ALEClassic


class AtariNet(nn.Module):
    """ Estimator used by DQN-style algorithms for ATARI games.
        Works with DQN, M-DQN and C51.
    """
    def __init__(self, action_no):
        super().__init__()

        self.action_no = out_size = action_no

        # get the feature extractor and fully connected layers
        self.__features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.__head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Linear(512, out_size),
        )

    def forward(self, x):
        assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float().div(255)

        x = self.__features(x)
        qs = self.__head(x.view(x.size(0), -1))
        
        return qs


def _load_checkpoint(fpath, device="cpu"):
    fpath = Path(fpath)
    with fpath.open("rb") as file:
        with GzipFile(fileobj=file) as inflated:
            return torch.load(inflated, map_location=device)


def _epsilon_greedy(state, model, eps=0.001):
    if torch.rand((1,)).item() < eps:
        return torch.randint(model.action_no, (1,)).item(), None
    q_val, argmax_a = model(state).max(1)
    return argmax_a.item(), q_val


def atari_data_generator(args):
    # game/seed/model
    ckpt_path = Path(args.path)
    game = ckpt_path.parts[-3]

    # set env
    ALE = ALEModern if "_modern/" in args.path else ALEClassic
    env = ALE(
        game,
        torch.randint(100_000, (1,)).item(),
        sdl=False, # Whether to show the gameplay
        device="cpu",
        clip_rewards_val=False,
    )

    # init model
    model = AtariNet(env.action_space.n)

    # load state
    ckpt = _load_checkpoint(args.path)
    model.load_state_dict(ckpt["estimator_state"])

    # configure policy
    policy = partial(_epsilon_greedy, model=model, eps=0.001)
    
    # Tuple for collection of data
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    current_run = list()

    while True:
        state, done = env.reset(), False
        while not done:
            action, _ = policy(state)
            next_state, reward, done, _ = env.step(action)
            
            state = state / 255. # Rescale from ints (0..255) to floats (0..1)
            state = state.reshape(4*84*84) # Delete first dimension, is always 1, and collapse all three remaining dimensions (channels, x_coordinate, y_coordinate)
            
            current_run.append(Transition(state, action, reward, done, next_state))
            
            state = next_state
        
        yield current_run
        
        current_run.clear()
        