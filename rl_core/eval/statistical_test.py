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
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping...")
            continue
        with open(path) as stream:
            steps.append(yaml.safe_load(stream)["global_steps"])
    return np.array(steps)

A = get_run_length("BufferPER")
B = get_run_length("Buffer")
print(f"A.mean={A.mean():.2f}, B.mean={B.mean():.2f}")

# Test if run lengths are normally distributed
# H₀: samples come from a normal distribution
# small p-value → reject normality
_, p = stats.normaltest(A)
print(f"normaltest A: {p=:.3f}")

# Compare means
# H₀: mean(A) >= mean(B)  (PER not faster)
# H₁: mean(A) < mean(B)   (PER faster)
# p-value = probability of observing a difference at least this extreme
# if the true means were equal
_, p = stats.ttest_ind(A, B, alternative="less", equal_var=False)
print(f"ttest_ind: {p=:.3f}")

# interpretation
# p < 0.05 → statistically significant
# p < 0.01 → strong evidence
# p < 0.001 → very strong evidence

effect = (A.mean() - B.mean()) / np.sqrt((A.var() + B.var())/2)
print("Cohen's d:", effect)

