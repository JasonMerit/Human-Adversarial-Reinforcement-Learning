import random
from functools import partial
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
import torch
import wandb
from torch import nn as nn
from rich import print
from torch.amp import GradScaler, autocast

from ..common import networks
from ..common.networks import RainbowTronNet
from ..common.replay_buffer import PrioritizedReplayBuffer

class Rainbow:
    buffer: PrioritizedReplayBuffer

    def __init__(self, envs, args: SimpleNamespace, device: torch.device) -> None:
        self.use_amp = args.use_amp
        self.device = device

        if not args.tron:
            net = networks.get_model()
            linear_layer = networks.FactorizedNoisyLinear
            self.q_policy = net(3, actions=3, linear_layer=linear_layer).to(device) # 3 channels (depth)
            self.q_target = net(3, actions=3, linear_layer=linear_layer).to(device) # 3 channels (depth)
        else:
            n_actions = envs.single_action_space.nvec[0]  # Either is fine (symmetric environment)
            obs_shape = envs.single_observation_space.shape[-3:]  # Ignore the stacked observations
            self.q_policy = RainbowTronNet(obs_shape, n_actions).to(device) # 3 channels (depth)
            self.q_target = RainbowTronNet(obs_shape, n_actions).to(device) # 3 channels (depth)
        self.q_target.load_state_dict(self.q_policy.state_dict())

        #k = 0
        #for parameter in self.q_policy.parameters():
        #    k += parameter.numel()
        #print(f'Q-Network has {k} parameters.')

        self.double_dqn = args.double_dqn

        self.buffer = PrioritizedReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.num_envs, use_amp=self.use_amp)
        self.n_step_gamma = args.gamma ** args.n_step

        self.max_grad_norm = args.max_grad_norm
        self.opt = torch.optim.Adam(self.q_policy.parameters(), lr=args.lr, eps=args.adam_eps)
        self.scaler = GradScaler(device=self.device, enabled=self.use_amp)

        loss_fn_cls = nn.MSELoss if args.loss_fn == 'mse' else nn.SmoothL1Loss
        self.loss_fn = loss_fn_cls(reduction=('none'))

    def sync_Q_target(self) -> None:
        self.q_target.load_state_dict(self.q_policy.state_dict())

    @torch.no_grad()
    def reset_noise(self, net) -> None:
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.reset_noise()

    @torch.no_grad()
    def disable_noise(self, net) -> None:
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.disable_noise()

    def act(self, states):
        """ computes an epsilon-greedy step with respect to the current policy self.q_policy """
        with torch.no_grad():
            with autocast(device_type=str(self.device), enabled=self.use_amp):
                action_values = self.q_policy(states, advantages_only=True)
                actions = torch.argmax(action_values, dim=1)
            return actions.cpu()

    @torch.no_grad()
    def td_target(self, reward: float, next_state, done: bool):
        self.reset_noise(self.q_target)
        if self.double_dqn:
            best_action = torch.argmax(self.q_policy(next_state, advantages_only=True), dim=1)
            next_Q = torch.gather(self.q_target(next_state), dim=1, index=best_action.unsqueeze(1)).squeeze()
            return reward + self.n_step_gamma * next_Q * (1 - done.float())
        else:
            max_q = torch.max(self.q_target(next_state), dim=1)[0]
            return reward + self.n_step_gamma * max_q * (1 - done.float())

    def train(self, batch_size, beta=None) -> Tuple[float, float, float]:
        indices, weights, (state, next_state, action, reward, done) = self.buffer.sample(batch_size, beta)
        weights = torch.from_numpy(weights).to(self.device)
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        # print(f"[yellow bold]State type: {type(state)}, State shape: {state.shape}, State device: {state.device}[/yellow bold]")
        # raise Exception("Debugging: check state type")

        self.opt.zero_grad()
        with autocast(device_type=str(self.device), enabled=self.use_amp):
            td_est = torch.gather(self.q_policy(state), dim=1, index=action.unsqueeze(1)).squeeze()
            td_tgt = self.td_target(reward, next_state, done)

            td_errors = td_est-td_tgt
            new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6  # 1e-6 is the epsilon in PER
            self.buffer.update_priorities(indices, new_priorities)

            losses = self.loss_fn(td_tgt, td_est)
            loss = torch.mean(weights * losses)

        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.opt)
        grad_norm = nn.utils.clip_grad_norm_(list(self.q_policy.parameters()), self.max_grad_norm)
        self.scaler.step(self.opt)
        self.scaler.update()

        return td_est.mean().item(), loss.item(), grad_norm.item()

    def save(self, path, verbose=True):
        torch.save(self.q_policy.state_dict(), path)
        if verbose:
            print(f"Model saved to {path}")
    
    @classmethod
    def from_checkpoint(cls, path, obs_shape, n_actions, device="cpu"):
        net = networks.get_model()
        linear_layer = networks.FactorizedNoisyLinear
        q_policy = net(obs_shape[0], actions=n_actions, linear_layer=linear_layer).to(device)
        q_policy.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        return q_policy