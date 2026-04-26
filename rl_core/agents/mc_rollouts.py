import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from rich import print

from .rainbow import RainbowAgent
from rl_core.env import PoLEnv
from rl_core.MCTS.vec_pol import VecPoLEnv
from .utils import TimerRegistry

def make_env(idx, size):
    def thunk():
        env = PoLEnv(size)
        # env = PoLEnv(size, idx == 0)  # Render first env
        env.action_space.seed(idx)
        return env
    return thunk


class MCTSAgent(RainbowAgent):
    """
    Extends RainbowAgent with Monte Carlo Rollout targets.
    Uses replay buffer states as rollout roots.
    """

    def __init__(self, obs_shape, n_actions, args, device, writer, name):
        super().__init__(obs_shape, n_actions, args, device, writer, name)

        self.rollout_depth = args.mcts_depth
        self.rollout_count = args.mcts_rollouts
        self.env = PoLEnv(args.size)
        self.envs = VecPoLEnv(args.batch_size, args.size)
        # self.envs = gym.vector.SyncVectorEnv([make_env(i, args.size) for i in range(args.batch_size)])

    @TimerRegistry.wrap_fn("mcts_learn")
    def learn(self):
        self.learning_steps += 1

        obs, actions, rewards, next_obs, dones, weights, indices = self.rb.sample(self.batch_size)
        
        # prediction by network
        # q = self.q_network(obs)  # [B, A]
        q = self.target_network(obs)  # [B, A]
        pred = q.gather(1, actions.long().to(self.device)).squeeze(1)

        # rollout target for those actions
        target_q = self._compute_rollout_targets_parallel(obs)
        # target_q = self._compute_rollout_targets(obs)
        target = target_q.gather(1, actions.long().to(self.device)).squeeze(1)

        # Loss and optimizer step
        loss_per_sample = F.smooth_l1_loss(pred, target, reduction="none")
        loss = (loss_per_sample * weights.squeeze()).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # optional logging hooks
        if self.writer and self.learning_steps % 100 == 0:
            self.writer.add_scalar(f"losses/{self.name}_rollout_td_loss", loss.item(), self.learning_steps)

    @TimerRegistry.wrap_fn("rollouts_parallel")
    def _compute_rollout_targets_parallel(self, obs_batch):
        """
        Parallelized version of rollout target computation.
        Returns:
            Q_rollout: [B, n_actions]
        """
        B = obs_batch.size(0)
        n_actions = self.q_network.adv_head[-1].out_features
        q_targets = torch.zeros((B, n_actions), device=self.device)

        for a in range(n_actions):
            returns = []

            for _ in range(self.rollout_count):
                returns.append(self._single_rollout_parallel(obs_batch, a))

            q_targets[:, a] = torch.stack(returns, dim=0).mean(dim=0)

        return q_targets
    
    @torch.no_grad()
    def _single_rollout_parallel(self, obs_batch, action):
        self.envs.set_state(obs_batch.cpu().numpy())
        
        actions = np.full((len(obs_batch),), action)
        obs, rewards, dones, _, _ = self.envs.step(actions)
        
        total_rewards = rewards.copy()
        alive = ~dones

        discount = 1.0
        for _ in range(self.rollout_depth):
            if not np.any(alive):
                break

            obs_alive = torch.from_numpy(obs[alive]).to(self.device)
            # obs_t = torch.from_numpy(obs).to(self.device)

            # actions[alive] = np.random.randint(4, size=alive.sum())
            actions[alive] = torch.argmax(self.q_network(obs_alive), dim=1).cpu().numpy()
            # actions = torch.argmax(self.target_network(obs_alive), dim=1).cpu().numpy()

            obs, rewards, dones, _, _ = self.envs.step(actions)

            discount *= self.gamma
            total_rewards[alive] += discount * rewards[alive]

            alive &= ~dones
            # print number of alive envs
            # print(f"Alive envs: {alive.sum()}, {actions}")

        return torch.from_numpy(total_rewards).to(self.device)
    
    @TimerRegistry.wrap_fn("rollouts")
    def _compute_rollout_targets(self, obs_batch):
        """
        Returns:
            Q_rollout: [B, n_actions]
        """

        B = obs_batch.size(0)
        n_actions = self.q_network.adv_head[-1].out_features
        q_targets = torch.zeros((B, n_actions), device=self.device)

        for b in range(B):
            state = obs_batch[b]

            for a in range(n_actions):
                returns = []

                for _ in range(self.rollout_count):
                    returns.append(self._single_rollout(state, a))

                q_targets[b, a] = sum(returns) / len(returns)

        return q_targets

    @torch.no_grad()
    def _single_rollout(self, state, action):
        # env = PoLEnv.from_state(state.cpu().numpy())
        self.env.set_state(state.cpu().numpy())

        obs, reward, done, _, _ = self.env.step(action)
        total_reward = reward

        discount = 1.0
        gamma = self.gamma

        steps = 0
        while not done and steps < self.rollout_depth:
            obs_t = torch.tensor(obs, device=self.device).unsqueeze(0)
            q = self.target_network(obs_t)

            action = torch.argmax(q, dim=1).item()
            obs, reward, done, _, _ = self.env.step(action)

            discount *= gamma
            total_reward += discount * reward

            steps += 1

        return total_reward