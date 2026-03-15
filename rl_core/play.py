from rl_core.tron_env.tron_env.env import TronEnv
from rl_core.tron_env.tron_env.wrappers import TronView

import torch
import gymnasium as gym

class TorchObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.device = device

    def observation(self, obs):
        return torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

def load_agent(agent_path, env):
    obs_shape = env.observation_space.shape  # (3, H, W)
    n_actions = env.action_space.n           # should be 3
    # Select the appropriate agent class based on the file name
    if "train" in agent_path.lower():
        from rl_core.agents.dqn import QNetwork
        return QNetwork.from_checkpoint(agent_path, obs_shape, n_actions)
    elif "dqn" in agent_path.lower():
        from rl_core.cleanrl.cleanrl.dqn import QNetwork as QNetwork
        return QNetwork.from_checkpoint(agent_path,obs_shape, n_actions)
    elif "ppo" in agent_path.lower():
        from rl_core.cleanrl.cleanrl.ppo import Agent
        return Agent.from_checkpoint(agent_path, obs_shape, n_actions)
    elif "rainbow" in agent_path.lower():
        from rl_core.cleanrl.cleanrl.rainbow import NoisyDuelingDistributionalNetwork
        return NoisyDuelingDistributionalNetwork.from_checkpoint(agent_path, obs_shape, n_actions)
    else:
        raise ValueError(f"Unknown agent type in path: {agent_path}")

def play(path):
    env = TronView(TronEnv())
    env = TorchObservationWrapper(env, device="cpu")
    state, _ = env.reset()

    agent = load_agent(path, env)
    agent.eval()  # Set the agent to evaluation mode
    
    while True:
        action = agent(state)  # Use the loaded model to select an action
        state, reward, done, _, info = env.step(action)

        if done:
            state, _ = env.reset()
            print(info.get("result"))

if __name__ == "__main__":
    # Get args
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    parser.add_argument("path", type=str, help="Path to the trained model checkpoint.")
    args = parser.parse_args()
    play(args.path)
    # play("runs/dqn_1/dqn.pth")
    # play("runs/ppo_cnn_881413/ppo_cnn.pth")