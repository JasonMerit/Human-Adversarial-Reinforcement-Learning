import torch, yaml
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from pathlib import Path
from rich import print

from rl_core.env import TronDuoEnv
from rl_core.env.wrappers import TorchObservationWrapper
from rl_core.eval.battle import make_agent

def collect_dataset(agent1, agent2, env, device):
    states, actions = [], []
    obs, info = env.reset()

    while True:
        a1, a2 = agent1.act(obs[:, 0]).item(), agent2.act(obs[:, 1]).item()
        states.append(obs[:, 1].squeeze(0))  # Remove player dimension
        actions.append(a2)

        obs, _, done, _, _ = env.step((a1, a2))
        if done:
            break

    # states = torch.tensor(states, dtype=torch.float32, device=device)
    states = torch.stack(states).to(device).float()
    actions = torch.tensor(actions, dtype=torch.long, device=device)

    return TensorDataset(states, actions)

def train_predictor(model, dataset, epochs=100, batch_size=32, lr=1e-3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1000 *epochs):
        for states, actions in loader:

            logits = model.predict(states)
            loss = F.cross_entropy(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch} loss {loss.item():.4f}")
        accuracy = test_predictor(model, dataset)
        print(f"epoch {epoch} accuracy {accuracy:.4f}")
        if accuracy > 0.99:
            print("Early stopping due to high accuracy")
            break

@torch.inference_mode()
def test_predictor(model, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    total, correct = 0, 0

    with torch.no_grad():
        for states, actions in loader:
            logits = model.predict(states)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == actions).sum().item()
            total += actions.size(0)

    return correct / total

def main(folder):
    """Train A.pth to predict B.pth"""
    folder = Path("runs") / folder
    with open(folder / "args.yml", "r") as f:
        args = yaml.safe_load(f)
        size = args['size']
    
    env = TorchObservationWrapper(TronDuoEnv(size=size), device="cpu")
    n_actions = env.unwrapped.n_actions
    obs_shape = env.unwrapped.obs_shape

    path1, path2 = folder / "A.pth", folder / "B.pth"
    agent1, agent2 = make_agent(path1, obs_shape, n_actions), make_agent(path2, obs_shape, n_actions)

    dataset = collect_dataset(agent1, agent2, env, device="cpu")

    train_predictor(agent1, dataset, epochs=100)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    parser.add_argument("folder", type=str, default="", help="Path folder of trained model checkpoints.")
    args = parser.parse_args()

    main(args.folder)
    