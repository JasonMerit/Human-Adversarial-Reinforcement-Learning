import math
import random
import torch
import torch.nn.functional as F

from rl_core import env


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}

        self.N = {}
        self.Q = {}
        self.M2 = {}

        self.N_i = {}
        self.N_j = {}

        self.total_visits = 0
        self.is_expanded = False


class SimultaneousMCTS:
    def __init__(self, env, q_net, action_space, gamma=0.99):
        self.env = env
        self.q_net = q_net
        self.A = action_space
        self.gamma = gamma

    def run(self, root_state, num_simulations=100):

        root = Node(root_state)

        for _ in range(num_simulations):

            env = self.env.clone()
            env.set_state(root_state)

            node = root
            path = []
            rewards = []

            while node.is_expanded:

                a_i = self.select_ucb1_tuned(node, player=0)
                a_j = self.select_ucb1_tuned(node, player=1)

                joint = (a_i, a_j)

                if joint not in node.children:
                    break

                path.append((node, joint))

                next_state, r_i, r_j, done = env.step(a_i, a_j)

                rewards.append(r_i)

                node = node.children[joint]

                if done:
                    break

            if not node.is_expanded:
                self.expand(node)

            value = self.evaluate(env, node.state)

            G = value

            for (node, joint), r in reversed(list(zip(path, rewards))):

                G = r + self.gamma * G

                self.update(node, joint, G)

        return self.extract_policy(root), self.root_value(root)

    def expand(self, node):

        node.is_expanded = True

        for a_i in range(self.A):
            node.N_i[a_i] = 0

        for a_j in range(self.A):
            node.N_j[a_j] = 0

        for a_i in range(self.A):
            for a_j in range(self.A):

                joint = (a_i, a_j)

                node.N[joint] = 0
                node.Q[joint] = 0.0
                node.M2[joint] = 0.0

    def evaluate(self, env, state):

        with torch.no_grad():
            q = self.q_net(state)
            v = torch.max(q).item()

        return v

    def variance(self, node, joint):

        n = node.N[joint]

        if n < 2:
            return 0

        return node.M2[joint] / (n - 1)

    def select_ucb1_tuned(self, node, player):

        best_action = None
        best_score = -1e9

        if player == 0:
            actions = node.N_i
        else:
            actions = node.N_j

        for a in actions:

            if player == 0:
                n = node.N_i[a]
            else:
                n = node.N_j[a]

            if n == 0:
                return a

            mean = self.marginal_mean(node, a, player)
            var = self.marginal_var(node, a, player)

            score = mean + math.sqrt(
                math.log(node.total_visits) / n *
                min(0.25, var + math.sqrt(2 * math.log(node.total_visits) / n))
            )

            if score > best_score:
                best_score = score
                best_action = a

        return best_action

    def marginal_mean(self, node, action, player):

        total = 0
        count = 0

        for (a_i, a_j), q in node.Q.items():

            if player == 0 and a_i == action:
                total += q * node.N[(a_i, a_j)]
                count += node.N[(a_i, a_j)]

            if player == 1 and a_j == action:
                total -= q * node.N[(a_i, a_j)]
                count += node.N[(a_i, a_j)]

        if count == 0:
            return 0

        return total / count

    def marginal_var(self, node, action, player):

        vals = []

        for (a_i, a_j), q in node.Q.items():

            if player == 0 and a_i == action:
                vals.append(q)

            if player == 1 and a_j == action:
                vals.append(-q)

        if len(vals) < 2:
            return 0

        mean = sum(vals) / len(vals)

        return sum((v - mean) ** 2 for v in vals) / len(vals)

    def update(self, node, joint, G):

        node.total_visits += 1

        node.N[joint] += 1
        n = node.N[joint]

        delta = G - node.Q[joint]
        node.Q[joint] += delta / n
        node.M2[joint] += delta * (G - node.Q[joint])

        a_i, a_j = joint

        node.N_i[a_i] += 1
        node.N_j[a_j] += 1

    def extract_policy(self, root):

        counts = torch.zeros(self.A)

        for a in range(self.A):
            counts[a] = root.N_i.get(a, 0)

        if counts.sum() == 0:
            return torch.ones(self.A) / self.A

        return counts / counts.sum()

    def root_value(self, root):

        total = 0
        weight = 0

        for joint in root.N:

            n = root.N[joint]
            q = root.Q[joint]

            total += n * q
            weight += n

        if weight == 0:
            return 0

        return total / weight


# Training Step
def train_step(q_net, optimizer, states, mcts, env):

    pis = []
    values = []

    for s in states:

        pi, v = mcts.run(s, num_simulations=100)

        pis.append(pi)
        values.append(v)

    pis = torch.stack(pis)
    values = torch.tensor(values)

    q = q_net(states)

    p = F.softmax(q, dim=1)

    v_pred = (p * q).sum(dim=1)

    policy_loss = -(pis * F.log_softmax(q, dim=1)).sum(dim=1).mean()

    value_loss = ((v_pred - values) ** 2).mean()

    loss = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



# Train Loop
def train_loop(env, q_net, replay_buffer, optimizer, training_steps=1000, batch_size=64):
    mcts = SimultaneousMCTS(env, q_net, action_space=env.num_actions)
    for step in range(training_steps):
        states = replay_buffer.sample(batch_size)
        loss = train_step(q_net, optimizer, states, mcts, env)