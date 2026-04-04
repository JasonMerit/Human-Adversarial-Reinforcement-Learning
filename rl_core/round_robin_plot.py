import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    win_matrix = np.load("rl_core/round_robin_results.npy")
    plt.figure(figsize=(8, 6))
    im = plt.imshow(win_matrix, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, label="Effective win rate")
    plt.title("Round Robin Winnings Heatmap (row wins vs column = 1.0)")
    plt.tight_layout()
    plt.show()
    