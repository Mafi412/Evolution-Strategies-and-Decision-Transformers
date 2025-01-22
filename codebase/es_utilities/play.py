# Functions for simulating and evaluating the model in the environment

from tqdm import trange

def simulate(wrapped_model, wrapped_environment, num_of_episodes):
    episode_returns, episode_lengths = list(), list()
    
    progress_bar = trange(num_of_episodes)
    for episode in progress_bar:
        episode_return, episode_length = 0., 0
        wrapped_model.reset_inner_state()
        state, done = wrapped_environment.reset(), False
        
        if wrapped_environment.timestep_limit is not None:
            for _ in range(wrapped_environment.timestep_limit):
                state, done, episode_return, episode_length = one_step(wrapped_model, wrapped_environment, state, episode_return, episode_length)

                if done:
                    break
                
        else:
            while not done:
                state, done, episode_return, episode_length = one_step(wrapped_model, wrapped_environment, state, episode_return, episode_length)
            
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        progress_bar.set_description(f"Episode {episode+1} - return (runtime): {episode_return} ({episode_length}) || " + \
            f"Mean return (mean runtime): {sum(episode_returns) / len(episode_returns)} ({sum(episode_lengths) // len(episode_lengths)})")
        
    return episode_returns, episode_lengths


def one_step(wrapped_model, wrapped_environment, state, episode_return, episode_length):
    action = wrapped_model.choose_action(state)

    next_state, reward, terminated, truncated = wrapped_environment.step(action)
    done = terminated or truncated

    wrapped_model.update_after_step(state, next_state, action, reward, terminated, truncated)
    
    episode_return += reward
    episode_length += 1
    
    return next_state, done, episode_return, episode_length
