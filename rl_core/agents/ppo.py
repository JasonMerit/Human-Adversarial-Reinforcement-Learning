from certifi.__main__ import args
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from torch import optim

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape

        # --- CNN Feature Extractor ---
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.cnn(dummy).shape[1]

        # --- Critic ---
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_flatten, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # --- Actor ---
        self.actor = nn.Sequential(
            layer_init(nn.Linear(n_flatten, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )
    
    def get_value(self, x):
        features = self.cnn(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.cnn(x)
        logits = self.actor(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(features)
    
    def forward(self, state):
        features = self.cnn(state)
        logits = self.actor(features)
        probs = Categorical(logits=logits)

        return probs.sample()
        
    @classmethod
    def from_checkpoint(cls, checkpoint_path, obs_shape, n_actions):
        agent = cls(obs_shape, n_actions)
        agent.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        return agent

class PPOAgent(nn.Module):
    def __init__(self, obs_shape, n_actions, device, args):
        super().__init__()
        self.network = ActorCriticNetwork(obs_shape, n_actions).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=args.learning_rate, eps=1e-5)
        self.args = args

    def get_values(self, x):
        return self.network.get_value(x)

    def get_action_and_value(self, x, action=None):
        return self.network.get_action_and_value(x, action)
        
    def learn(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values):
        # Optimizing the policy and value network
        b_inds = np.arange(self.args.batch_size)
        clipfracs = []
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                break
