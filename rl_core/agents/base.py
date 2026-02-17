from rl_core.utils.helper import bcolors

class Agent:
    def bind_env(self, env):
        if getattr(self, "_bound", False):
            return
        self._check_env(env)
        self._bound = True

    def _check_env(self, env):
        raise NotImplementedError(f"{bcolors.OKCYAN}Subclasses should implement this method{bcolors.ENDC}")
    
    def reset(self, seed=None):
        raise NotImplementedError(f"{bcolors.OKCYAN}Subclasses should implement this method{bcolors.ENDC}")

class RandomAgent(Agent):
    def __call__(self, state):
        return self.action_space.sample()
    
    def reset(self, seed=None):
        pass
    
    def _check_env(self, env):
        self.action_space = env.action_space