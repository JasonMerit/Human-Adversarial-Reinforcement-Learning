# ===================================
# === Vision Grid & State Encoder ===
# ===================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

def make_vision_grid(grid, player_pos, opp_pos, player_head, size=5):
    """Extract 5x5 vision grid from agent's POV [paper Fig 2]"""
    half = size // 2
    vh, vw = player_head
    # Crop around player head
    v_player = grid[vh-half:vh+half+1, vw-half:vw+half+1]  # Player trails
    v_opp = grid[vh-half:vh+half+1, vw-half:vw+half+1]     # Opp trails  
    v_wall = np.zeros((size, size))  # Wall grid (boundaries)
    v_opp_head = np.zeros((size, size))  # Opp head relative
    
    # Binary encode (1=occupied)
    v_player = (v_player > 0).astype(float)
    v_opp = (v_opp > 0).astype(float)
    
    # Walls at edges
    v_wall[0, :] = v_wall[-1, :] = v_wall[:, 0] = v_wall[:, -1] = 1.0
    
    # Relative opp head
    oh_rel_h = opp_pos[vh-half:vh+half+1, vw-half:vw+half+1]
    v_opp_head[oh_rel_h == 1] = 1.0
    
    return np.concatenate([v_player.flatten(), v_opp.flatten(), 
                          v_wall.flatten(), v_opp_head.flatten()])  # 75 nodes

class TronMLP(nn.Module):
    def __init__(self, input_dim=75, hidden_dim=200):  # Large VG optimal
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU(beta=0.01)  # Key: ELU > sigmoid [paper]
        self.q_head = nn.Linear(hidden_dim, 4)      # Q(s,a1)..Q(s,a4)
        self.opp_head = nn.Linear(hidden_dim, 4)    # P(s,o1)..P(s,o4)

    def forward(self, x):
        h = self.elu(self.fc1(x))
        q_vals = self.q_head(h)
        opp_logits = self.opp_head(h)
        return q_vals, F.softmax(opp_logits, -1)
    

# ===================================
# ======= Agent & Training ==========
# ===================================

class TronAgent:
    def __init__(self, env, lr=0.005, gamma=0.95, hidden_dim=200):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = TronMLP(hidden_dim=hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.target_net = None  # Use online net for simplicity
        self.rollout_horizon = 20  # n-step MC
        self.rollouts_per_action = 10  # m=10 best
        
    def get_state(self, obs):
        """Vision grid from env obs [paper Sec 4.1]"""
        grid = obs['grid']  # 10x10 visited
        player_pos = obs['player_pos']
        opp_pos = obs['opp_pos']
        player_head = np.argwhere(player_pos)[0]  # (h,w)
        return torch.FloatTensor(make_vision_grid(grid, player_pos, opp_pos, player_head)).unsqueeze(0).to(self.device)
    
    def act(self, obs, epsilon=0.0):
        state = self.get_state(obs)
        q_vals, opp_policy = self.net(state)
        
        if random.random() < epsilon:
            return random.randrange(4)
        
        # MC rollouts with opp model [Algorithm 1]
        rollout_qs = self._mc_rollouts(obs, q_vals[0].cpu().numpy())
        return rollout_qs.argmax()
    
    def _mc_rollouts(self, obs, q_vals, num_rollouts=10):
        """Monte Carlo rollouts using learned opp model [paper Sec 4.2]"""
        rollout_rewards = np.zeros(4)
        
        for action in range(4):
            for _ in range(num_rollouts):
                reward = self._single_rollout(obs, action)
                rollout_rewards[action] += reward
        
        return rollout_rewards / num_rollouts
    
    def _single_rollout(self, obs, start_action):
        """Simulate n-step rollout [Alg 1]"""
        sim_obs = obs.copy()
        step = 0
        
        # Start action
        sim_obs = self.env.step(sim_obs, start_action)  # Your env sim step
        if sim_obs['done']:
            return self.gamma**step * sim_obs['reward']
        
        while step < self.rollout_horizon and not sim_obs['done']:
            state_t = self.get_state(sim_obs)
            _, opp_policy_t = self.net(state_t)
            
            # Greedy self action
            q_t, _ = self.net(state_t)
            self_action = q_t.argmax().item()
            
            # Sample opp action from model
            opp_action = np.random.choice(4, p=opp_policy_t[0].cpu().numpy())
            
            sim_obs = self.env.step(sim_obs, self_action, opp_action)
            step += 1
            
            if sim_obs['done']:
                return self.gamma**step * sim_obs['reward']
        
        # Truncated: final Q
        final_q, _ = self.net(self.get_state(sim_obs))
        return self.gamma**step * final_q.max().item()
    
    def update(self, state, action, reward, next_state, done, opp_action):
        """Multi-task loss: Q + opponent CE [paper Sec 4.2]"""
        state = state.to(self.device)
        q_vals, opp_policy = self.net(state)
        
        if done:
            q_target = reward
        else:
            next_q, _ = self.net(next_state)
            q_target = reward + self.gamma * next_q.max()
        
        # Q-loss
        q_loss = F.mse_loss(q_vals[0, action], q_target)
        
        # Opponent modeling loss
        opp_target = F.one_hot(torch.tensor(opp_action), 4).float()
        opp_loss = F.cross_entropy(self.net[1], opp_target)  # From opp_logits
        
        loss = q_loss + 0.5 * opp_loss  # Balance [implicit in paper]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()



# ===================================
# ========== Training Loop ==========
# ===================================

def train(env, episodes=1_500_000):
    agent = TronAgent(env)
    scores = []
    
    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        
        while not obs['done']:
            # Epsilon decay: 10% → 0% over first 750k games
            epsilon = max(0, 0.1 * (1 - ep / 750_000))
            action = agent.act(obs, epsilon)
            
            next_obs, reward, done, opp_action = env.step(action)
            
            state = agent.get_state(obs)
            next_state = agent.get_state(next_obs)
            
            loss = agent.update(state, action, reward, next_state, done, opp_action)
            
            obs = next_obs
            total_reward += reward
            
        scores.append(total_reward)
        
        if ep % 10000 == 0:
            print(f"Ep {ep}: Score {np.mean(scores[-10000:]):.3f}")
    
    return agent

