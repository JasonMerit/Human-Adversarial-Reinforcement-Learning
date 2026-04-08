"""
This file handles parsing and validation of the cli arguments to the train_rainbow.py file.
If left unspecified, some argument defaults are set dynamically here.
"""

import argparse
import distutils
import socket
from copy import deepcopy

import random

def read_args():
    parse_bool = lambda b: bool(distutils.util.strtobool(b))
    parser = argparse.ArgumentParser(description='Training framework for Rainbow DQN\n'
                                                 '  - supports environments from the ALE (via gym), gym-retro and procgen\n'
                                                 '  - individial components of Rainbow can be adjusted with cli args (below)\n'
                                                 '  - uses vectorized environments and batches environment steps for best performance\n'
                                                 '  - uses the large IMPALA-CNN (with 2x channels by default)',
                                     formatter_class=argparse.RawTextHelpFormatter)

    # training settings
    parser.add_argument('--training_frames', type=int, default=10_000_000, help='train for n environment interactions ("game_frames" in the code)')
    parser.add_argument('--record_every', type=int, default=60*50, help='wait at least x seconds between episode recordings (default is to use environment specific presets)')
    parser.add_argument('--seed', type=int, default=-1, help='seed for pytorch, numpy, environments, random')
    parser.add_argument('--use_wandb', type=parse_bool, default=True, help='whether use "weights & biases" for tracking metrics, video recordings and model checkpoints')
    parser.add_argument('--use_amp', type=parse_bool, default=True, help='whether to enable automatic mixed precision for the forward passes')
    parser.add_argument('--der', type=parse_bool, default=False, help='enable data-efficient-rainbow profile (overrides some of the settings below)')
    parser.add_argument('--decorr', type=parse_bool, default=True, help='try to decorrelate state/progress in parallel envs')

    # environment settings
    parser.add_argument('--env_name', type=str, default='Tron',
                        help='the gym/procgen/retro environment name, should be either gym:[name], retro:[name] or procgen:[name]\n'
                             'some gym envs:   MsPacman, Phoenix, Breakout, Qbert, Amidar, SpaceInvaders, Assault\n'
                             'some retro envs: SuperMarioWorld-Snes, MortalKombat3-Genesis, SpaceMegaforce-Snes, SmashTV-Nes, AirBuster-Genesis, NewZealandStory-Genesis, Paperboy-Nes\n'
                             'progcen envs:    bigfish, bossfight, caveflyer, chaser, climber, coinrun, dodgeball, fruitbot, heist, jumper, leaper, maze, miner, ninja, plunder, starpilot')
    parser.add_argument('--procgen_distribution_mode', type=str, default='hard',
                        help='what variant of the procgen levels to use, the options are "easy", "hard", "extreme", "memory", "exploration". All'
                             ' games support "easy" and "hard", while other options are game-specific. The default is "hard". Switching to "easy'
                             '" will reduce the number of timesteps required to solve each game and is useful for testing or when working with l'
                             'imited compute resources.')
    parser.add_argument('--retro_state', type=str, default='default', help='initial gym-retro state name or "default" or "randomized" (to randomize on episode reset)')
    parser.add_argument('--time_limit', type=int, default=108_000, help='environment time limit for gym & retro (in non-frameskipped native env frames)')
    parser.add_argument('--wandb_tag', type=str, default=None, help='')

    # dqn settings
    parser.add_argument('--buffer_size', type=int, default=int(2 ** 20), help='capacity of experience replay buffer (must be a power of two)')
    parser.add_argument('--burnin', type=int, default=80_000, help='how many transitions should be in the buffer before start of training')
    parser.add_argument('--gamma', type=float, default=.99, help='reward discount factor')
    parser.add_argument('--sync_dqn_target_every', type=int, default=32_000, help='sync Q target net every n frames')

    parser.add_argument('--batch_size', type=int, default=256, help='sample size when sampling from the replay buffer')
    parser.add_argument('--parallel_envs', type=int, default=64, help='number of envs in the vectorized env')
    parser.add_argument('--train_count', type=int, default=2, help='how often to train on a batch_size batch for every step (of the vectorized env)')

    # rainbow settings
    parser.add_argument('--double_dqn', type=parse_bool, default=True, help='whether to use the double-dqn TD-target')
    parser.add_argument('--prioritized_er', type=parse_bool, default=True, help='whether to use prioritized experience replay')
    parser.add_argument('--prioritized_er_beta0', type=float, default=0.45, help='importance sampling exponent for PER (0.4 for rainbow, 0.5 for dopamine)')
    parser.add_argument('--prioritized_er_time', type=int, default=None, help='time period over which to increase the IS exponent (+inf for dopamine; default is value of training_frames)')
    parser.add_argument('--n_step', type=int, default=3, help='the n in n-step bootstrapping')
    parser.add_argument('--noisy_dqn', type=parse_bool, default=True, help='whether to use noisy nets dqn')
    parser.add_argument('--noisy_sigma0', type=float, default=100, help='sigma_0 parameter for noisy nets dqn')

    # optimizer settings
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate for adam (0.0000625 for rainbow paper/dopamine, 0.00025 for DQN/procgen paper)')
    parser.add_argument('--adam_eps', type=float, default=None, help='epsilon for adam (0.00015 for rainbow paper/dopamine, 0.0003125 for DQN/procgen paper); default is to use 0.005/batch_size')
    parser.add_argument('--max_grad_norm', type=float, default=10, help='gradient will be clipped to ensure its l2-norm is less than this')
    parser.add_argument('--loss_fn', type=str, default='huber', help='loss function ("mse" or "huber")')
    args = parser.parse_args()

    # some initial checks to ensure all arguments are valid
    assert (args.sync_dqn_target_every % args.parallel_envs) == 0 # otherwise target may not be synced since the main loop iterates in steps of parallel_envs
    assert args.loss_fn in ('mse', 'huber')
    assert args.burnin > args.batch_size

    args.seed = random.randint(0, 1e6) if args.seed == -1 else args.seed

    # apply default values if user did not specify custom settings
    if args.adam_eps is None: args.adam_eps = 0.005/args.batch_size
    if args.prioritized_er_time is None: args.prioritized_er_time = args.training_frames

    # hyperparameters for DER are adapted from https://github.com/Kaixhin/Rainbow
    if args.der:
        args.parallel_envs = 1
        args.batch_size = 32
        args.train_count = 1
        args.burnin = 6400
        args.n_step = 20
        args.sync_dqn_target_every = 8000
        args.buffer_size = 2 ** 17
        args.lr = 0.00025
        args.adam_eps = 0.0003125

    # clean up the parameters that get logged to wandb
    args.instance = socket.gethostname()
    wandb_log_config = deepcopy(vars(args))
    wandb_log_config['env_type'] = args.env_name[:args.env_name.find(':')]
    del wandb_log_config['record_every']
    del wandb_log_config['use_wandb']
    if not args.env_name.startswith('retro:'):
        for k in list(wandb_log_config.keys()):
            if k.startswith('retro'):
                del wandb_log_config[k]
    if not args.env_name.startswith('procgen:'):
        for k in list(wandb_log_config.keys()):
            if k.startswith('procgen'):
                del wandb_log_config[k]
    del wandb_log_config['wandb_tag']

    return args, wandb_log_config
