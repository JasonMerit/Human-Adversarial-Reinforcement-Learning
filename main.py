# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
import os
from pathlib import Path
import yaml
import numpy as np
from rich import print
import gymnasium as gym

from rl_core.env import PoLEnv

import numpy as np
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory


def worker(name, shape):
    shm = SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

    for _ in range(5):
        arr += 1

    shm.close()


if __name__ == "__main__":

    shape = (4,)

    shm = SharedMemory(create=True, size=np.zeros(shape, dtype=np.float32).nbytes)

    shared_array = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
    shared_array[:] = 0

    p1 = Process(target=worker, args=(shm.name, shape))
    p2 = Process(target=worker, args=(shm.name, shape))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print(shared_array)

    shm.close()
    shm.unlink()