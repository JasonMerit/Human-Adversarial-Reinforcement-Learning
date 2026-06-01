import os
from dataclasses import dataclass
import random, yaml
from rich import print

import tyro

@dataclass
class Args:
    exp_name: str = "Tron_Rainbow"
    """the name of this experiment"""
    seed: int = -1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    # total_timesteps: int = 400_000  # Less time steps for Knegt due to long time in MC_ROLLOUTS
    total_timesteps: int = 2_000_000
    # total_timesteps: int = 4_000_000# 1_0000_000  # 1_000_000 ~ 1 hour on HPC
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4#0.0000625
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    buffer_size: int = 50_000#500_000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 1000#8000
    """the timesteps it takes to update the target network"""
    batch_size: int = 128#32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.2#0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 1000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    mirror_prob: float = 0.1
    """the probability of mirroring the state in the replay buffer for data augmentation (specific to Tron)"""

    # NN architecture
    conv1: int = 16
    """the number of channels for the 1st conv layer (specific to Tron)"""
    conv2: int = 32
    """the number of channels for the 2nd conv layer (specific to Tron)"""
    hidden_size: int = 32
    """the hidden size of the neural networks"""

    # Prioritized replay buffer
    per: bool = True
    """whether to use a prioritized experience replay buffer"""
    per_alpha: float = 0.5
    """alpha parameter for prioritized replay buffer"""
    per_beta: float = 0.4
    """beta parameter for prioritized replay buffer"""
    per_eps: float = 1e-6
    """epsilon parameter for prioritized replay buffer"""


    # MCTS
    mcts: bool = False
    """whether to use MCTS agent"""
    rollouts: int = 64
    """the number of rollouts to perform in MCTS"""
    mcts_c: float = 1.0
    """the exploration constant for MCTS"""
    horizon: int = 10
    """the maximum depth for MCTS"""

    # Player modeller
    seq_len = 5
    """the context window length for the player modeler, i.e. how many past states and actions to consider when predicting the opponent's next action"""
    emb_dim = 16  # Rando
    """the embedding dimension for the player modeler"""
    hidden = 32  # Rando
    """the hidden dimension for the player modeler"""
    pm_lr = 1e-3
    """the learning rate for the player modeler"""

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
    pol: bool = False
    """Whether to use the proof of learning environment"""
    vec: bool = False
    """Whether to use vectorized environments (only applicable for PoL)"""
    knegt: bool = False
    """Whether to use the KnegtAgent, which adds opponent modeling to DQN"""

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
        # args.total_timesteps = 400
        # args.render = True
        # args.buffer_size *= 1e89
        args.buffer_size = args.batch_size * 10
        args.batch_size = 2
        args.conv1 = 64
        args.conv2 = 64
        args.hidden = 64

        args.exp_name = "debug_" + args.exp_name
        # args.pol = True  # Test PoL environment in debug mode
    
    # assert args.num_envs > args.train_frequency, "num_envs should be greater than train_frequency for correct training logic"
    assert not args.save or (args.save and args.track), "If save is true, track must also be true"
    # assert not (args.dqn and args.mcts), "Cannot specify both dqn and mcts agents"

    return args

def load_args(path, verbose=True):
    assert os.path.exists(path), f"File not found: {path}"
    args = Args()
    
    with open(path, "r") as f:
        loaded_args = yaml.safe_load(f)

    skipped = []
    for key, value in loaded_args.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            skipped.append(key)
    if verbose and skipped:
        print(f"{path} : Skipped unknown {skipped}")
    return args