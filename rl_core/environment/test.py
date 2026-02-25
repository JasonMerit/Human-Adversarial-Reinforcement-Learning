import yaml

from .env import TronDualEnv, TronSingleEnv
from .wrappers import TronView
from rl_core.agents import HeuristicAgent

with open("rl_core/config.yml", "r") as f:
    config = yaml.safe_load(f)
# single = config.get("single", True)
size = tuple(config.get("grid"))

adv = HeuristicAgent()
env = TronSingleEnv(adv, size)
env = TronView(env, fps=5, scale=50)
adv.bind_env(env)

state, _ = env.reset()
last_action = 1
while True:
    action = env.action_space.sample()
    while (action + 2) % 4 == last_action:
        action = env.action_space.sample()
    
    state, reward, done, _, _ = env.step(action)
    last_action = action

    if done:
        state, _ = env.reset()



