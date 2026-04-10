# Based on https://ymd_h.gitlab.io/cpprb/examples/dqn_per/
import cpprb
import torch
from .utils import TimerRegistry

class PrioritizedReplayBuffer:

    def __init__(self, args, device):
        self.device = device
        self.batch_size = args.batch_size
        self.beta = args.prioritized_replay_beta
        self.beta_step = (1.0 - args.prioritized_replay_beta) / args.total_timesteps

        env_dict = {"obs": {"shape": (3, 25, 25)}, "act": {}, "rew": {}, "next_obs": {"shape": (3, 25, 25)}, "done": {}}
        Nstep = {"size": args.n_step, "rew": "rew", "next": "next_obs"}
        self._rb = cpprb.PrioritizedReplayBuffer(args.batch_size * 100, env_dict, Nstep=Nstep, alpha=args.prioritized_replay_alpha, eps=args.prioritized_replay_eps)

    @TimerRegistry.wrap_fn("buffer_add")
    def add(self, obs, act, rew, next_obs, done):
        for i in range(len(obs)):
            self._rb.add(obs=obs[i], act=act[i], rew=rew[i], next_obs=next_obs[i], done=done[i])
            if done[i]: self._rb.on_episode_end()

    @TimerRegistry.wrap_fn("buffer_sample")
    def sample(self):
        sample = self._rb.sample(self.batch_size, beta=self.beta)
        self.beta = min(1.0, self.beta + self.beta_step)  # Should probs be constant later

        obs_t    = torch.as_tensor(sample["obs"], device=self.device)
        next_obs = torch.as_tensor(sample["next_obs"], device=self.device)
        actions  = torch.as_tensor(sample["act"], device=self.device, dtype=torch.long)
        rewards  = torch.as_tensor(sample["rew"], device=self.device)
        dones    = torch.as_tensor(sample["done"], device=self.device)
        weights  = torch.as_tensor(sample["weights"], device=self.device)
        indices  = sample["indexes"]

        return obs_t, actions, rewards, next_obs, dones, weights, indices

    @TimerRegistry.wrap_fn("buffer_update")
    def update(self, indices, priorities):
        self._rb.update_priorities(indices, priorities)