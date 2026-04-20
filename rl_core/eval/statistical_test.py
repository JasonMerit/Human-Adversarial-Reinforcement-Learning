# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
import os
from pathlib import Path
import yaml
import numpy as np
from rich import print
from scipy import stats

def get_run_length(folder):
    folders = [f for f in os.listdir("runs") if f.startswith(folder + "_")]
    steps = []
    for f in folders:
        path = Path("runs") / f / "results.yml"
        # Read steps_taken from file and add to toal_steps
        # Check if exists
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping...")
            continue
        with open(path) as stream:
            steps.append(yaml.safe_load(stream)["global_steps"])
    return np.array(steps)

A = get_run_length("BufferPER")
B = get_run_length("Buffer")
print(f"A.mean={A.mean():.2f}, B.mean={B.mean():.2f}")

# H₀: A is normal
# p-val = p(> chi-squared statistic) 
# small p-val unlikely H₀
_, p = stats.normaltest(A)
print(f"normaltest: {p=:.3f}")

# H₀: A is not faster than B (A >= B)
# H₁: A is faster than B (A < B)
# If p(A) = p(B), then p = p(a>=b) just by chance
# if low chance, reject H₀ and accept H₁

# _, p = stats.mannwhitneyu(A, B, alternative="less", method="exact")
# print(f"mannwhitneyu: {p=:.3f}")
# p < 0.05 → significant
# p < 0.01 → strong evidence
# p < 0.001 → very strong evidence

_, p = stats.ttest_ind(A, B, alternative="less", method=None)
# _, p = stats.ttest_ind(A, B, alternative="less", method=None)
print(f"ttest_ind: {p=:.3f}")

