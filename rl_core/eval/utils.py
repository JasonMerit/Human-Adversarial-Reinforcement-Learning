
def make_dqn(path, obs_shape, n_actions):
    from rl_core.agents.dqn import QNetwork
    return QNetwork.from_checkpoint(path, obs_shape, n_actions, device="cpu")

def make_rainbow(path, obs_shape, n_actions):
    from rl_core.agents.rainbow import DuelingNetwork
    return DuelingNetwork.from_checkpoint(path, obs_shape, n_actions, device="cpu")

