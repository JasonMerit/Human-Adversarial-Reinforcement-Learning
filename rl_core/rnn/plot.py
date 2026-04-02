import numpy as np

# Load data from npy
history = np.load("rl_core/rnn/rnn_reward_history.npy")
print(max(history))
import matplotlib.pyplot as plt
plt.plot(history)
plt.xlabel("Episode (x10)")

plt.ylabel("Average Reward per Step")
plt.title("RNN Training Reward History")
plt.show()
