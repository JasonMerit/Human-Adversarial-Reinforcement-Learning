# Based on https://ymd_h.gitlab.io/cpprb/examples/dqn_per/
import cpprb
import torch
import numpy as np
from .utils import TimerRegistry

class ReplayBuffer:

    def __init__(self, args, device):
        self.device = device
        self.batch_size = args.batch_size
        env_dict = {
            "obs": {"shape": (3, 25, 25), "dtype": np.uint8}, 
            "act": {"dtype": np.uint8}, 
            "rew": {"dtype": np.uint8}, 
            "next_obs": {"shape": (3, 25, 25), "dtype": np.uint8}, 
            "done": {"dtype": np.bool_}
            }
        self._rb = cpprb.ReplayBuffer(args.buffer_size, env_dict)

    @TimerRegistry.wrap_fn("buffer_add")
    def add(self, obs, act, rew, next_obs, done):
        self._rb.add(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)

    @TimerRegistry.wrap_fn("buffer_sample")
    def sample(self):
        sample = self._rb.sample(self.batch_size)

        obs_t    = torch.as_tensor(sample["obs"], device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(sample["next_obs"], device=self.device, dtype=torch.float32)
        actions  = torch.as_tensor(sample["act"], device=self.device, dtype=torch.long)
        rewards  = torch.as_tensor(sample["rew"], device=self.device)
        dones    = torch.as_tensor(sample["done"], device=self.device)
        weights  = torch.ones_like(rewards, device=self.device)  # Uniform weights for non-prioritized buffer
        indices  = None  # No indices needed for non-prioritized buffer

        return obs_t, actions, rewards, next_obs, dones, weights, indices

    def update(self, indices, priorities):
        pass

class PrioritizedReplayBuffer:

    def __init__(self, args, device):
        self.device = device
        self.batch_size = args.batch_size
        self.beta = args.prioritized_replay_beta

        total_train_steps = (args.total_timesteps // args.num_envs) * (args.num_envs // args.train_frequency)
        self.beta_step = (1.0 - args.prioritized_replay_beta) / total_train_steps  # Linear annealing of beta over the course of training

        env_dict = {
            "obs": {"shape": (3, 25, 25), "dtype": np.uint8}, 
            "act": {"dtype": np.uint8}, 
            "rew": {"dtype": np.uint8}, 
            "next_obs": {"shape": (3, 25, 25), "dtype": np.uint8}, 
            "done": {"dtype": np.bool_}
            }
        self._rb = cpprb.PrioritizedReplayBuffer(args.buffer_size, env_dict, alpha=args.prioritized_replay_alpha, eps=args.prioritized_replay_eps)

    @TimerRegistry.wrap_fn("buffer_add")
    def add(self, obs, act, rew, next_obs, done):
        self._rb.add(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)
            

    @TimerRegistry.wrap_fn("buffer_sample")
    def sample(self):
        sample = self._rb.sample(self.batch_size, beta=self.beta)
        self.beta = min(1.0, self.beta + self.beta_step)  # Anneal beta towards 1.0

        obs_t    = torch.as_tensor(sample["obs"], device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(sample["next_obs"], device=self.device, dtype=torch.float32)
        actions  = torch.as_tensor(sample["act"], device=self.device, dtype=torch.long)
        rewards  = torch.as_tensor(sample["rew"], device=self.device)
        dones    = torch.as_tensor(sample["done"], device=self.device)
        weights  = torch.as_tensor(sample["weights"], device=self.device)
        indices  = sample["indexes"]

        return obs_t, actions, rewards, next_obs, dones, weights, indices

    @TimerRegistry.wrap_fn("buffer_update")
    def update(self, indices, priorities):
        self._rb.update_priorities(indices, priorities)

if __name__ == "__main__":
    from .argp import read_args
    import numpy as np
    args = read_args()
    rb = ReplayBuffer(args, "cpu")

    for i in range(4):
        obs = np.random.randn(args.num_envs, 3, 25, 25)
        act = np.zeros(args.num_envs, dtype=int)
        rew = np.zeros(args.num_envs)
        next_obs = np.random.randn(args.num_envs, 3, 25, 25)
        done = np.zeros(args.num_envs, dtype=bool)
        rb.add(obs, act, rew, next_obs, done)

    obs_t, actions, rewards, next_obs, dones, weights, indices = rb.sample()
    print(obs_t.shape, actions.shape, rewards.shape, next_obs.shape, dones.shape, weights.shape)