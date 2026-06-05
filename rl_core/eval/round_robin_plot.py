import numpy as np
import matplotlib.pyplot as plt

win_matrix = np.load("rl_core/eval/rr_results/NN.npy")
print(win_matrix)

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(win_matrix, cmap="RdYlGn", vmin=0, vmax=1)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Match score")

ax.set_title("Round-Robin Team Tournament")
ax.set_xlabel("Opponent Team")
ax.set_ylabel("Team")

# Hide ticks
# ax.set_xticks([])
# ax.set_yticks([])
n = win_matrix.shape[0]
ax.set_xticks(range(n))
ax.set_yticks(range(n))

ax.set_xticklabels(range(n))
ax.set_yticklabels(range(n))


plt.tight_layout()
plt.show()