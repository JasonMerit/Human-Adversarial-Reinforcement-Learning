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
        env = TronDualEnv(size)


    territorry = False

    if territorry:
        state, _ = env.reset()
        sv.view_heuristic(state, get_territories(*state))
        while True:
            action = TronView.wait_for_keypress() if single else TronView.wait_for_both_inputs()
            state, reward, done, _, info = env.step(action)

            if done:
                state, _ = env.reset()
                print(f"{reward=}")
            sv.view_heuristic(state, get_territories(*state))
            TronView.wait_for_keypress()
    
    else:
        from rl_core.utils.heuristics import voronoi

        coords1 = [(1, 3), (2, 2), (3, 3), (2, 2), (1, 3), (2, 0), (2, 0), (1, 3), (0, 2), (0, 2), (0, 3), (1, 2), (0, 1), (0, 1), (1, 1)]
        scores1 = [0, 0, -36, -22, -22, -21, -32, -38, -27, 1, -9, 27, 29, 32, 29]
        arr = np.array([[coord, score] for coord, score in zip(coords1, scores1)], dtype=object)

        coords2 = [(1, 3), (1, 3), (2, 2), (2, 2), (1, 2), (2, 3), (1, 0), (0, 0), (0, 0), (3, 0), (0, 3), (0, 0), (3, 1), (0, 0), (1, 3), (0, 0), (3, 1), (3, 1), (3, 1), (2, 2), (3, 1), (2, 2)]
        scores2 = [0, 0, 0, 0, 31, -22, 12, -48, -46, -50, -46, -18, -1, -1, -42, -8, 1, 1, 1, 1, 1, -20]
        arr2 = np.array([[coord, score] for coord, score in zip(coords2, scores2)], dtype=object)

        state, _ = env.reset()
        sv.view_heuristic(state, get_territories(*state))

        indx = 0
        while True:
            action = arr[indx][0] 
            state, _, _, _, _ = env.step(action)
            
            print(voronoi(*state), " ~ ", arr[indx][1])

            sv.view_heuristic(state, get_territories(*state))
            indx += 1
            if indx == len(arr2):  # arr2 is longer
                quit()
            if indx == len(arr):
                arr = arr2
                state, _ = env.reset()
                indx = 0
                print("============")
                
            TronView.wait_for_keypress()




