import collections
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
# Generate the data just on CPU (the model is too small and the copying from CPU to GPU and vice versa would be unnecessarily costly)
try:
    tf.config.set_visible_devices([], "GPU")
except:
    pass
import tensorflow_probability as tfp


class Actor(tf.keras.Model):
    def __init__(self, env):
        super().__init__()
        
        self.env = env
    
        self._hidden_layer_1 = tf.keras.layers.Dense(
            256,
            activation=tf.keras.activations.relu
        )
        self._layer_norm_1 = tf.keras.layers.LayerNormalization(
            scale=False
        )
        self._hidden_layer_2 = tf.keras.layers.Dense(
            256,
            activation=tf.keras.activations.relu
        )
        self._layer_norm_2 = tf.keras.layers.LayerNormalization(
            scale=False
        )
        
        self._means_layer = tf.keras.layers.Dense(env.action_space.shape[0])
        self._sds_layer = tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.keras.activations.exponential)
        
        self._log_alpha = tf.Variable(np.log(0.1), dtype=tf.float32)

                
    def call(self, inputs, sample = False):
        hidden = self._hidden_layer_1(inputs)
        hidden = self._layer_norm_1(hidden)
        
        hidden = self._hidden_layer_2(hidden)
        hidden = self._layer_norm_2(hidden)
        
        mus = self._means_layer(hidden)
        sds = self._sds_layer(hidden)
        sds = tf.clip_by_value(sds, 1e-8, 1e8)
        
        actions_distribution = tfp.distributions.Normal(mus, sds if sample else 0)
        actions_distribution = tfp.bijectors.Tanh()(actions_distribution)
        actions_distribution = tfp.bijectors.Scale((self.env.action_space.high - self.env.action_space.low) / 2)(actions_distribution)
        actions_distribution = tfp.bijectors.Shift((self.env.action_space.high + self.env.action_space.low) / 2)(actions_distribution)
        
        actions = actions_distribution.sample()
        
        log_probs = tf.reduce_mean(actions_distribution.log_prob(actions), axis=-1)
        
        alpha = tf.math.exp(self._log_alpha)
        
        return actions, log_probs, alpha


def humanoid_data_generator(args):
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    
    # Create the environment
    env = gym.make("Humanoid-v4")
    
    # Load agent
    actor = Actor(env)
    actor(np.array([env.observation_space.sample()]), sample=True)
    actor.load_weights(args.model_path)
    
    # Create the asynchroneous vector environment for data collection.
    venv = gym.vector.make("Humanoid-v4", args.envs, asynchronous=True)

    batches = [list() for _ in range(args.envs)]
    
    # Tuple for collection of data
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    states = venv.reset(seed=args.seed)[0]
    while True:
        action, _, _ = actor(states)

        next_states, rewards, terminated, truncated, _ = venv.step(action)
        dones = terminated | truncated
        for i in range(args.envs):
            batches[i].append(Transition(states[i], action[i], rewards[i] if not terminated[i] else 0, dones[i], next_states[i]))
            
        states = next_states
        
        for i in range(args.envs):
            if dones[i]:
                yield batches[i]
                
                batches[i].clear()
