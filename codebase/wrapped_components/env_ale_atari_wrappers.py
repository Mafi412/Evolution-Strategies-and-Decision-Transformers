from es_utilities.wrappers import EsEnvironmentWrapper


class ALEAtariWrapper(EsEnvironmentWrapper):
    def __init__(self, env, seed):
        super().__init__(env, seed)
        
    def reset(self):
        state = self.env.reset()
        return state
    
    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done, done
    
    def set_seed(self, seed):
        if seed is not None:
            self.env.ale.setInt("random_seed", seed)
    
    @property
    def state_shape(self):
        return (1, self.env.window, 84, 84)
    
    @property
    def timestep_limit(self):
        return 108000
