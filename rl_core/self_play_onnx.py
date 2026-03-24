from rl_core.tron_env.tron_env.env import TronDuoEnv
from rl_core.tron_env.tron_env.wrappers import TronView

import onnxruntime as ort
import gymnasium as gym
import numpy as np

class ONNXObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # (players, channels, height, width) -> (players, height, width, channels)
        # (2, 3, 25, 25) -> (2, 25, 25, 3) 
        return np.transpose(obs, (0, 2, 3, 1))
    
    
def play():
    # env = TronDuoEnv()
    env = TronView(TronDuoEnv(), fps=100)
    env = ONNXObsWrapper(env)
    obs, _ = env.reset()

    human = ort.InferenceSession("tron_unity/Assets/human.onnx")
    adversary = ort.InferenceSession("tron_unity/Assets/adversary.onnx")    

    while True:
        obs0, obs1 = obs
        obs0 = np.expand_dims(obs0, axis=0).astype(np.float32)  # (1, H, W, C)
        obs1 = np.expand_dims(obs1, axis=0).astype(np.float32)
        obs0 = {human.get_inputs()[0].name: obs0}
        obs1 = {adversary.get_inputs()[0].name: obs1}
        
        a0 = human.run(None, obs0)[0].argmax()
        a1 = adversary.run(None, obs1)[0].argmax()

        obs, reward, done, _, info = env.step([a0, a1])

        if done:
            import time
            time.sleep(100)
            obs, _ = env.reset()
            print(info.get("result"))

if __name__ == "__main__":
    play()