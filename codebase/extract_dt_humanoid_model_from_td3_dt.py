from argparse import ArgumentParser
from pathlib import Path

import torch

from td3_dt.dt_for_td3 import DecisionTransformerTD3Policy, LastStateExtractor, DecisionTransformerActor

from stable_baselines3 import TD3


def main(args):
    from_path = Path(args.path_to_the_saved_td3_model)
    
    # Patch TD3 to avoid allocating the replay buffer
    orig_init_replay_buffer = TD3._setup_model
    
    class DummyReplayBuffer:
        def __init__(*args, **kwargs): pass
        def add(*args, **kwargs): pass
        def sample(*args, **kwargs): raise NotImplementedError
        def extend(*args, **kwargs): pass
        def __len__(self): return 0

    def no_replay_buffer(self):
        self.replay_buffer = None
        orig_replay_buffer_class = getattr(self, "replay_buffer_class", None)
        self.replay_buffer_class = DummyReplayBuffer
        try:
            orig_init_replay_buffer(self)
        finally:
            if orig_replay_buffer_class is not None:
                self.replay_buffer_class = orig_replay_buffer_class
       
    TD3._setup_model = no_replay_buffer

    model = TD3.load(from_path, custom_objects={
        "DecisionTransformerTD3Policy": DecisionTransformerTD3Policy,
        "LastStateExtractor": LastStateExtractor,
        "DecisionTransformerActor": DecisionTransformerActor,
    })

    # Restore original method to avoid side effects
    TD3._setup_model = orig_init_replay_buffer
    
    to_path = Path(args.path_where_to_save_the_dt_model)
    if to_path.suffix != ".model":
        to_path = to_path.with_suffix(to_path.suffix + ".model")
    
    torch.save(model.policy.actor.dt_model.state_dict(), to_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path_to_the_saved_td3_model", type=str, help="Path to the saved TD3 model.")
    parser.add_argument("path_where_to_save_the_dt_model", type=str, help="Path where to save the extracted DT model.")
    
    main(parser.parse_args())
