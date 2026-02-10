from agents.base import Agent

class RandomAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, state):
        return self.action_space.sample()
    
    def _check_env(self, env):
        pass