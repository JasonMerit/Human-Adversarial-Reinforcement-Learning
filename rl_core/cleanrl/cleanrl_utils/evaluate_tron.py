import torch
import gymnasium as gym

env = gym.make("Tron-v0")

def evaluate_policy(model, device):
    obs, _ = env.reset()
    model.eval()  # Set the model to evaluation mode
    while True:
        with torch.no_grad():
            action = model(torch.tensor(obs).unsqueeze(0).to(device))
        obs, reward, done, _, info = env.step(action)
        
        if done:
            model.train()  # Set the model back to training mode
            return info.get("result") == 2  # Bike2 wins