import numpy as np
from rl_core.utils.heuristics import voronoi
from gymnasium import spaces
from rl_core.utils.helper import bcolors
from .base import Agent

class HeuristicAgent(Agent):
    """
    Tron agent that selects moves by evaluating the chamber_heuristic
    for all possible actions and picking the one with highest score.
    """
    def reset(self, seed=None):
        pass

    def __call__(self, state : tuple):
        """
        Returns the action that maximizes chamber_heuristic.
        """
        trails, you, other = state

        best_score = -np.inf
        best_action = None
        for action, dir in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            new_pos = you + dir

            if not(0 <= new_pos[0] < trails.shape[1]) or \
               not(0 <= new_pos[1] < trails.shape[0]) or \
               trails[new_pos[1], new_pos[0]] != 0:
                continue

            score = voronoi(trails, new_pos, other)

            if score > best_score:
                best_score = score
                best_action = action
            
        if best_action is None:
            return 0  # If no valid moves, just pick up (or any default)
        return best_action
    
    def _check_env(self, env):
        if not isinstance(env.observation_space, spaces.Tuple):
            raise ValueError(f"{bcolors.FAIL}HeuristicAgent requires an environment with a tuple observation space. Try removing wrappers.{bcolors.ENDC}")    
    

if __name__ == "__main__":
    import yaml
    from rl_core.environment.env import TronDualEnv, TronSingleEnv
    from rl_core.environment.wrappers import TronView
    from rl_core.agents import HeuristicAgent
    from rl_core.utils import StateViewer
    from rl_core.utils.heuristics import get_territories

    with open("rl_core/config.yml", "r") as f:
        config = yaml.safe_load(f)
    single = config.get("single", True)
    size = tuple(config.get("grid"))

    sv = StateViewer(size, scale=50, fps=1, single=True)

    if single:
        env = TronSingleEnv(HeuristicAgent(), size)
    else:
        adv = HeuristicAgent()
        env = TronDualEnv(size)

    # env = TronView(env, fps=5, scale=50)

    state, _ = env.reset()
    sv.view_heuristic(state, get_territories(*state))
    while True:
        action = TronView.wait_for_keypress() if single else TronView.wait_for_both_inputs()
        state, reward, done, _, info = env.step(action)

        if done:
            state, _ = env.reset()
            print(f"{reward=}")
        sv.view_heuristic(state, get_territories(*state))




