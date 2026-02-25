import yaml

from .env import TronDualEnv, TronSingleEnv
from .wrappers import TronView
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




