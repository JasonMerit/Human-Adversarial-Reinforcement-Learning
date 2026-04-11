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

    total_timesteps: int = 10_000#10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0000625
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 8000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    n_step: int = 3
    """the number of steps to look ahead for n-step Q learning"""
    prioritized_replay_alpha: float = 0.5
    """alpha parameter for prioritized replay buffer"""
    prioritized_replay_beta: float = 0.4
    """beta parameter for prioritized replay buffer"""
    prioritized_replay_eps: float = 1e-6
    """epsilon parameter for prioritized replay buffer"""
    n_atoms: int = 51
    """the number of atoms"""
    v_min: float = -10
    """the return lower bound"""
    v_max: float = 10
    """the return upper bound"""

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

def read_args():
    args = tyro.cli(Args)

    if args.seed == -1:
        args.seed = random.randint(0, 1e6)

    if args.debug:
        args.learning_starts = args.batch_size
        args.save = False
        args.num_envs = 5
        # args.total_checkpoints = 1
        args.total_timesteps = args.learning_starts * args.num_envs + 2
    
    assert args.num_envs > args.train_frequency, "num_envs should be greater than train_frequency for correct training logic"
    assert not args.save or (args.save and args.track), "If save is true, track must also be true"

    return args