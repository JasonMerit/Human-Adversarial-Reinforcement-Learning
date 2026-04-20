"""
This file defines all the neural network architectures available to use.
"""
from functools import partial
from math import sqrt

import torch
from torch import nn as nn, Tensor
from torch.nn import init
import torch.nn.functional as F

class FactorizedNoisyLinear(nn.Module):
    """ The factorized Gaussian noise layer for noisy-nets dqn. """
    def __init__(self, in_features: int, out_features: int, sigma_0: float = .5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0  # .5 from paper appendix

        # weight: w = \mu^w + \sigma^w . \epsilon^w
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        # bias: b = \mu^b + \sigma^b . \epsilon^b
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in
        scale = 1 / sqrt(self.in_features)

        init.uniform_(self.weight_mu, -scale, scale)
        init.uniform_(self.bias_mu, -scale, scale)

        init.constant_(self.weight_sigma, self.sigma_0 * scale)
        init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        # f(x) = sgn(x)sqrt(|x|)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        # like in eq 10 and 11 of the paper
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        # y = wx + d, where
        # w = \mu^w + \sigma^w * \epsilon^w
        # b = \mu^b + \sigma^b * \epsilon^b
        return F.linear(input,
                        self.weight_mu + self.weight_sigma*self.weight_epsilon,
                        self.bias_mu + self.bias_sigma*self.bias_epsilon)

# class FactorizedNoisyLinear(nn.Module):
#     """ Factorized Gaussian noise layer for per-env NoisyNets """
#     def __init__(self, in_features: int, out_features: int, sigma_0: float = 0.5):
#     # def __init__(self, in_features: int, out_features: int, sigma_0: float = 0.5, num_envs: int = 5):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.sigma_0 = sigma_0
#         # self.num_envs = num_envs  # number of parallel envs / agents

#         # Base parameters
#         self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
#         self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
#         self.bias_mu = nn.Parameter(torch.empty(out_features))
#         self.bias_sigma = nn.Parameter(torch.empty(out_features))

#         # Noise buffers: one set per env
#         self.register_buffer('weight_epsilon', torch.empty(num_envs, out_features, in_features))
#         self.register_buffer('bias_epsilon', torch.empty(num_envs, out_features))

#         self.reset_parameters()
#         self.reset_noise()  # initialize

#     @torch.no_grad()
#     def reset_parameters(self):
#         scale = 1 / sqrt(self.in_features)
#         init.uniform_(self.weight_mu, -scale, scale)
#         init.uniform_(self.bias_mu, -scale, scale)
#         init.constant_(self.weight_sigma, self.sigma_0 * scale)
#         init.constant_(self.bias_sigma, self.sigma_0 * scale)

#     @torch.no_grad()
#     def _get_noise(self, size: int):
#         noise = torch.randn(size, device=self.weight_mu.device)
#         return noise.sign().mul_(noise.abs().sqrt_())

#     @torch.no_grad()
#     def reset_noise(self):
#         """Reset noise for all envs independently"""
#         for env_idx in range(self.num_envs):
#             epsilon_in = self._get_noise(self.in_features)
#             epsilon_out = self._get_noise(self.out_features)
#             self.weight_epsilon[env_idx].copy_(epsilon_out.outer(epsilon_in))
#             self.bias_epsilon[env_idx].copy_(epsilon_out)

#     @torch.no_grad()
#     def disable_noise(self):
#         self.weight_epsilon.zero_()
#         self.bias_epsilon.zero_()

#     def forward(self, input: Tensor):
#         """
#         Expects input shape: (num_envs, batch_per_env, in_features)
#         or (num_envs, in_features) if batch_size=1 per env.
#         Uses the corresponding noise for each env.
#         """
#         if input.dim() == 2 and self.num_envs > 1:
#             # assume input shape is (num_envs, in_features)
#             out = torch.stack([
#                 F.linear(input[i],
#                          self.weight_mu + self.weight_sigma * self.weight_epsilon[i],
#                          self.bias_mu + self.bias_sigma * self.bias_epsilon[i])
#                 for i in range(self.num_envs)
#             ])
#             return out
#         else:
#             # fallback for single env or normal linear
#             return F.linear(input,
#                             self.weight_mu + self.weight_sigma * self.weight_epsilon[0],
#                             self.bias_mu + self.bias_sigma * self.bias_epsilon[0])

class Dueling(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, value_branch, advantage_branch):
        super().__init__()
        self.flatten = nn.Flatten()
        self.value_branch = value_branch
        self.advantage_branch = advantage_branch

    def forward(self, x, advantages_only=False):
        x = self.flatten(x)
        advantages = self.advantage_branch(x)
        if advantages_only:
            return advantages

        value = self.value_branch(x)
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))


class DuelingAlt(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, l1, l2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            l1,
            nn.ReLU(),
            l2
        )

    def forward(self, x, advantages_only=False):
        res = self.main(x)
        advantages = res[:, 1:]
        value = res[:, 0:1]
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))

class NatureCNN(nn.Module):
    """
    This is the CNN that was introduced in Mnih et al. (2013) and then used in a lot of later work such as
    Mnih et al. (2015) and the Rainbow paper. This implementation only works with a frame resolution of 84x84.
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            linear_layer(3136, 512),
            nn.ReLU(),
            linear_layer(512, actions),
        )

    def forward(self, x, advantages_only=None):
        return self.main(x)


class DuelingNatureCNN(nn.Module):
    """
    Implementation of the dueling architecture introduced in Wang et al. (2015).
    This implementation only works with a frame resolution of 84x84.
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.dueling = Dueling(
                nn.Sequential(linear_layer(3136, 512),
                              nn.ReLU(),
                              linear_layer(512, 1)),
                nn.Sequential(linear_layer(3136, 512),
                              nn.ReLU(),
                              linear_layer(512, actions))
            )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        return self.dueling(f, advantages_only=advantages_only)


class ImpalaCNNSmall(nn.Module):
    """
    Implementation of the small variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((6, 6))

        self.dueling = Dueling(
                nn.Sequential(linear_layer(1152, 256),
                              nn.ReLU(),
                              linear_layer(256, 1)),
                nn.Sequential(linear_layer(1152, 256),
                              nn.ReLU(),
                              linear_layer(256, actions))
            )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        f = self.pool(f)
        return self.dueling(f, advantages_only=advantages_only)


class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth, norm_func):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv_0 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_ = self.conv_0(self.relu(x))
        x_ = self.conv_1(self.relu(x_))
        return x+x_

class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x


class ImpalaCNNLarge(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, linear_layer, model_size=1, spectral_norm=False):
        super().__init__()

        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=norm_func),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=norm_func),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func_last),
            nn.ReLU()
        )

        # self.pool = torch.nn.AdaptiveMaxPool2d((4, 4))

        with torch.no_grad():
            dummy = torch.zeros(1, in_depth, 25, 25)
            f = self.main(dummy)
            # f = self.pool(f)
            n_flatten = f.view(1,-1).size(1)

        self.dueling = Dueling(
            nn.Sequential(linear_layer(1024, 256),
                          nn.ReLU(),
                          linear_layer(256, 1)),
            nn.Sequential(linear_layer(1024, 256),
                          nn.ReLU(),
                          linear_layer(256, actions)))

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        # f = self.pool(f)
        return self.dueling(f, advantages_only=advantages_only)


# def get_model(model_str, spectral_norm):
#     if model_str == 'nature': return NatureCNN
#     elif model_str == 'dueling': return DuelingNatureCNN
#     elif model_str == 'impala_small': return ImpalaCNNSmall
#     elif model_str.startswith('impala_large:'):
#         return partial(ImpalaCNNLarge, model_size=int(model_str[13:]), spectral_norm=spectral_norm)

class RainbowTronNet(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, c, h, w)).shape[1]

        self.value = nn.Sequential(
            nn.Linear(n_flat, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(n_flat, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x, advantages_only=False):
        f = self.cnn(x)
        a = self.advantage(f)
        if advantages_only:
            return a
        v = self.value(f)
        return v + (a - a.mean(dim=1, keepdim=True))

def get_model():
    return partial(ImpalaCNNLarge, model_size=2, spectral_norm="all")