import math
import copy
import torch
import torch.nn.functional as F


############################################
# Node
############################################

class Node:

    def __init__(self, state, parent=None):

        self.state = state
        self.parent = parent

        self.children = {}

        self.N = {}
        self.W = {}
        self.Q = {}
        self.P = {}

        self.total_visits = 0
        self.expanded = False


############################################
# AlphaZero MCTS
############################################

class AlphaZeroMCTS:

    def __init__(self, env, q_net, opponent_net, num_actions, gamma=0.99, c_puct=1.5):

        self.env = env
        self.q_net = q_net
        self.opponent_net = opponent_net

        self.A = num_actions
        self.gamma = gamma
        self.c_puct = c_puct


    ############################################
    # Run search
    ############################################

    def run(self, root_state, num_simulations=100):

        root = Node(root_state)

        self.expand(root)

        for _ in range(num_simulations):

            env = self.env.clone()
            env.set_state(root_state)

            node = root
            path = []
            rewards = []

            while node.expanded:

                action = self.select(node)

                path.append((node, action))

                opponent_action = self.sample_opponent(node.state)

                next_state, reward, done = env.step(action, opponent_action)

                rewards.append(reward)

                if action not in node.children:
                    child = Node(next_state, parent=node)
                    node.children[action] = child
                else:
                    child = node.children[action]

                node = child

                if done:
                    value = reward
                    break

                if not node.expanded:
                    value = self.expand(node)
                    break

            G = value

            for (node, action), r in reversed(list(zip(path, rewards))):

                G = r + self.gamma * G

                node.N[action] += 1
                node.W[action] += G
                node.Q[action] = node.W[action] / node.N[action]

                node.total_visits += 1

        return self.extract_policy(root), self.root_value(root)


    ############################################
    # Expand node
    ############################################

    def expand(self, node):

        with torch.no_grad():

            q = self.q_net(node.state)

            p = F.softmax(q, dim=0)

            value = torch.max(q).item()

        for a in range(self.A):

            node.P[a] = p[a].item()
            node.N[a] = 0
            node.W[a] = 0
            node.Q[a] = 0

        node.expanded = True

        return value


    ############################################
    # Selection
    ############################################

    def select(self, node):

        best_score = -1e9
        best_action = None

        for a in range(self.A):

            q = node.Q[a]

            u = self.c_puct * node.P[a] * math.sqrt(node.total_visits + 1) / (1 + node.N[a])

            score = q + u

            if score > best_score:
                best_score = score
                best_action = a

        return best_action


    ############################################
    # Opponent policy
    ############################################

    def sample_opponent(self, state):

        with torch.no_grad():

            q = self.opponent_net(state)

            p = F.softmax(q, dim=0)

        return torch.multinomial(p, 1).item()


    ############################################
    # Extract policy
    ############################################

    def extract_policy(self, root):

        counts = torch.tensor([root.N[a] for a in range(self.A)], dtype=torch.float32)

        if counts.sum() == 0:
            return torch.ones(self.A) / self.A

        return counts / counts.sum()


    ############################################
    # Root value
    ############################################

    def root_value(self, root):

        total = 0
        visits = 0

        for a in range(self.A):

            total += root.Q[a] * root.N[a]
            visits += root.N[a]

        if visits == 0:
            return 0

        return total / visits



############################################
# Training step
############################################

def train_step(q_net, optimizer, states, mcts, env):

    policies = []
    values = []

    for s in states:

        pi, v = mcts.run(s, num_simulations=100)

        policies.append(pi)
        values.append(v)

    policies = torch.stack(policies)
    values = torch.tensor(values)

    q = q_net(states)

    p = F.softmax(q, dim=1)

    v_pred = (p * q).sum(dim=1)

    policy_loss = -(policies * F.log_softmax(q, dim=1)).sum(dim=1).mean()

    value_loss = ((v_pred - values) ** 2).mean()

    loss = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



############################################
# Training loop
############################################

def train(env, q_net, replay_buffer, optimizer, steps):

    for step in range(steps):

        opponent_net = copy.deepcopy(q_net)

        mcts = AlphaZeroMCTS(
            env,
            q_net,
            opponent_net,
            env.num_actions
        )

        states = replay_buffer.sample(32)

        loss = train_step(q_net, optimizer, states, mcts, env)

        print("step", step, "loss", loss)