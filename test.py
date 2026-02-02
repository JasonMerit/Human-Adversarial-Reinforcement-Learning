import numpy as np

# A = np.array([1, 2])
# A += np.array([3, 1])
# # Random 10x10 array
# B = np.random.rand(5, 5)
# print(A)
# print(B)
# print(B[A])
# print(B[tuple(A)])

A = np.array([[1, 0], [0, 0]])
print(A)
B = np.rot90(A, k=1, axes=(1, 0))
print(B)