# OpenAI-ES (on a Decision Transformer)

This project contains codebase for our implementation of a distributed algorithm from the class of evolution strategies, [OpenAI-ES](https://arxiv.org/abs/1703.03864) (*es* folder), as well as means to analyze the data gathered during the runs.

There are also a few scripts running experiments to test those algorithms in a MuJoCo Humanoid locomotion environment with a simple feed-forward model and with a [Decision Transformer](https://arxiv.org/abs/2106.01345) architecture as the agent's policy. Additionaly, there is a script running an experiment in Atari games environment with the Decision transformer.

For the Decision Transformer, there is a possibility to use a pretrained model to seed the search. Pretrained models can be found in folder *pretrained_ckpts*. Code used for their training can be found in *supervised_pretraining* folder.

## How-to:

### Start a training

The provided training scripts (*train_\*.py*) provide a showcase of how the training script should look like. The argumens are described in the scripts or You can see a help by running the scripts with *-h* or *--help* option.

The underlying algorithm implementation requires MPI for their parallelization, so they need to be run using an MPI launcher.

### Simulate a trained agent

There are scripts for simulating the agent in the environment (*play_*.py*). For Humanoid environment, there can either be no visual output, just the return obtained and runtime of the epsiode; or there can be a video-recording of the agent's rollouts; or there can be a classical visual output. Atari environment currently does not support the recording of the rollouts. The provided scripts stand as a showcases of how to implement a custom replay script.

### Perform a basic data analysis

A script *plot_experiments.py* is provided which plots the data collected during the run of the given algorithm.

### Utilize a custom ...

#### Agent policy architecture

In *es_utilities* folder, *wrappers.py* file, there is an **EsModelWrapper** class, which You need to derive from. The resulting class should then override the non-implemented class functions. The custom policy will be stored in a field *model*. There is even a possibility to utilize a Virtual Batch Normalization, provided by the wrapper. Examples may be found in folder *wrapped_components*.

#### Environment

In *es_utilities* folder, *wrappers.py* file, there is an **EsEnvironmentWrapper** class, which You need to derive from. The resulting class should then override the non-implemented class functions, and even the *state_shape* property, in case the current implemantation would not return the desired state shape of the given environment. The custom environment will be stored in a field *env*. Examples may again be found in folder *wrapped_components*.