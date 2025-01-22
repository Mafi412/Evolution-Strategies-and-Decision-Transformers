from es_utilities.wrappers import EsEnvironmentWrapper


class GymMujocoWrapper(EsEnvironmentWrapper):
    def __init__(self, env, seed, reward_scale):
        super().__init__(env, seed)
        self.reward_scale = reward_scale
        
    def reset(self):
        state, _ = self.env.reset()
        return state
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        return next_state, reward / self.reward_scale, terminated, truncated
    
    def set_seed(self, seed):
        self.env.reset(seed=seed)
    
    @property
    def timestep_limit(self):
        return 1000
