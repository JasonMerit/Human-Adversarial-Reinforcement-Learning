import os
from dataclasses import dataclass
import random

import tyro

@dataclass
class Args:
    exp_name: str = "Tron_Rainbow"
    """the name of this experiment"""
    seed: int = -1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    total_timesteps: int = 1_0000_000  # 1_000_000 ~ 1 hour on HPC
    """total timesteps of the experiments"""
    learning_rate: float = 0.0000625
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 8000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.1
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.80
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

    # Prioritized replay buffer
    per: bool = True
    """whether to use a prioritized experience replay buffer"""
    prioritized_replay_alpha: float = 0.5
    """alpha parameter for prioritized replay buffer"""
    prioritized_replay_beta: float = 0.4
    """beta parameter for prioritized replay buffer"""
    prioritized_replay_eps: float = 1e-6
    """epsilon parameter for prioritized replay buffer"""

    # Distributional
    c51: bool = False
    """whether to use the C51 distributional RL algorithm"""
    n_atoms: int = 51
    """the number of atoms"""
    v_min: float = -100  # R_min / (1 - gamma)
    """the return lower bound"""
    v_max: float = 100
    """the return upper bound"""

    # Noisy Nets
    noisy: bool = False
    """whether to use noisy linear layers"""

    # Jason's additions
    save: bool = True
    """whether to save the final model"""
    track: bool = True
    """whether to track the experiment with Tensorboard and save hyperparameters and results to `runs{exp_name}`"""
    total_checkpoints: int = 10
    """the number of checkpoints to save (computed in runtime)"""
    debug: bool = False
    """if true, will manually set a few values to make the code run faster for debugging purposes"""
    tron: bool = True
    """if true, will use the Tron-specific network architecture"""
    render: bool = False
    """if true, will render the 1st environment"""
    size: int = 25
    """set square size of Tron"""

    # Set by HPC
    hpc: bool = False
    job_name: str = ""
    job_index: str = ""    

def read_args():
    args = tyro.cli(Args)

    if args.seed == -1:
        args.seed = random.randint(0, 1e6)

    if args.debug:
        args.save = args.track = False
        args.num_envs = 5
        # args.total_checkpoints = 1
        args.total_timesteps = 400
        args.render = True
        # args.buffer_size *= 1e89
        args.buffer_size = args.batch_size * 10

        args.exp_name = "debug_" + args.exp_name
    
    assert args.num_envs > args.train_frequency, "num_envs should be greater than train_frequency for correct training logic"
    assert not args.save or (args.save and args.track), "If save is true, track must also be true"

    return args