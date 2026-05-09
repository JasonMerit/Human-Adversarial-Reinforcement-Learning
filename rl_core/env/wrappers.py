import gymnasium as gym
import numpy as np
import torch

from . import utils, TronDuoEnv, TronEnvBase
from .tron import Tron, Result

class TronView(gym.Wrapper):

    blue = (34, 49, 63)
    blue_alt = (41, 64, 82)
    green = (20, 180, 20)
    green_alt = (20, 220, 20)
    red = (180, 20, 20)
    red_alt = (220, 20, 20)

    @property
    def state(self):
        return self.env.unwrapped.state
    
    @property
    def n_actions(self):
        return self.env.unwrapped.n_actions


    def __init__(self, env, fps=10, scale=20):
        super().__init__(env)
        import pygame
        self.tron = self.env.unwrapped.tron
        
        self.pg = pygame
        self.pg.init()

        self.scale = scale
        size = self.tron.size
        self.window_size = (size * self.scale, size * self.scale)
        self.screen = self.pg.display.set_mode(self.window_size)
        self.trails_screen = self.pg.Surface((size, size), flags=self.pg.SRCALPHA)

        background = self.pg.Surface((size, size))
        for x in range(size):
            for y in range(size):
                color = self.blue if (x + y) % 2 else self.blue_alt
                background.set_at((x, y), color)
        self.background = self.pg.transform.scale(background, self.window_size)

        self.pg.display.set_caption("Tron Game")
        
        self.clock = self.pg.time.Clock()
        self.fps = fps        

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)

        self.screen.blit(self.background, (0, 0))
        self.trails_screen.fill((0, 0, 0, 0))  # Clear trails with transparency

        # Fill in walls (neccesary after first move is made before env starts)
        walls = self.tron.walls
        for y, row in enumerate(walls):
            for x, cell in enumerate(row):
                if cell == 0: continue
                color = self.green if cell == 1 else self.red
                self.trails_screen.set_at((x, y), color)
                
        self._render()
    
        self.prev1 = self.tron.pos1.copy()
        self.prev2 = self.tron.pos2.copy()
        return state, info
    
    def step(self, action):
        result = self.env.step(action)
        
        self.screen.blit(self.background, (0, 0))
        self.trails_screen.set_at((self.tron.pos1[0], self.tron.pos1[1]), self.green_alt)
        self.trails_screen.set_at((self.tron.pos2[0], self.tron.pos2[1]), self.red_alt)
        self.trails_screen.set_at((self.prev1[0], self.prev1[1]), self.green)
        self.trails_screen.set_at((self.prev2[0], self.prev2[1]), self.red)
        self._render()

        self.prev1 = self.tron.pos1.copy()
        self.prev2 = self.tron.pos2.copy()

        # Input
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                self.pg.quit()
                exit()
            if event.type == self.pg.KEYDOWN:
                if event.key == self.pg.K_q or event.key == self.pg.K_ESCAPE:
                    self.pg.quit()
                    exit()
                
        self.clock.tick(self.fps)
        
        return result

    def _render(self):
        self.screen.blit(self.pg.transform.scale(self.trails_screen, self.window_size), (0, 0))
        self.pg.display.flip()
    


    @staticmethod
    def wait_for_keypress():
        import pygame as pg
        clock = pg.time.Clock()
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit()
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_q or event.key == pg.K_ESCAPE:
                        pg.quit()
                        exit()
                    if event.key == pg.K_SPACE:
                        return
                    if event.key == pg.K_LEFT:
                        return 0
                    if event.key == pg.K_RIGHT:
                        return 2
                    if event.key == pg.K_UP:
                        return 1
                    # WASD is absolute coordinates
                    if event.key == pg.K_w:
                        return 0
                    if event.key == pg.K_d:
                        return 1
                    if event.key == pg.K_s:
                        return 2
                    if event.key == pg.K_a:
                        return 3

            clock.tick(30)

    @staticmethod
    def wait_for_both_inputs():
        import pygame as pg
        clock = pg.time.Clock()
        p1_dir = None
        p2_dir = None

        player1_keys = [pg.K_w, pg.K_d, pg.K_s, pg.K_a]
        player2_keys = [pg.K_UP, pg.K_RIGHT, pg.K_DOWN, pg.K_LEFT]

        while p1_dir is None or p2_dir is None:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit()
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_q or event.key == pg.K_ESCAPE:
                        pg.quit()
                        exit()
                    if event.key in player1_keys:
                        p1_dir = player1_keys.index(event.key)
                    elif event.key in player2_keys:
                        p2_dir = player2_keys.index(event.key)
                
                clock.tick(30)

        return p1_dir, p2_dir


class TronPlay(TronDuoEnv):
    """Take control of the first agent and play against the provided opponent policy"""

    def __init__(self,  policy: callable, size=25, render=False):
        super().__init__(size, render)
        self.policy = policy
        import pygame
        self.pg = pygame
    
    def _process_input(self):
        for event in self.pg.event.get():
            if event.type == self.pg.KEYDOWN:
                if event.key in [self.pg.K_q, self.pg.K_ESCAPE]:
                    self.pg.quit()
                    exit()
                if event.key in [self.pg.K_UP, self.pg.K_w]:
                    self.heading1 = 0
                elif event.key in [self.pg.K_LEFT, self.pg.K_a]:
                    self.heading1 = 3
                elif event.key in [self.pg.K_RIGHT, self.pg.K_d]:
                    self.heading1 = 1
                elif event.key in [self.pg.K_DOWN, self.pg.K_s]:
                    self.heading1 = 2

    def step(self, kek=0):
        self._process_input()

        obs = TronDuoEnv.encode(self.state)[1]  # Get opponent's observation (the second one)
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        opp_action = self.policy.act(obs)  # Get opponent action based on their observation
        self.heading2 = (self.heading2 + (opp_action - 1)) % 4  # Because (left, forward, right)
        
        dir1 = TronEnvBase.action_mapping[self.heading1]
        dir2 = TronEnvBase.action_mapping[self.heading2]
    
        result = self.tron.tick(dir1, dir2)
        done = result != Result.PLAYING
        obs = TronDuoEnv.encode(self.state)
        reward = self.reward_dict[result]

        if self.render:
            self.view()

        return done

# class TronPlay(gym.Wrapper):
#     """Take control of the first agent and play against the provided opponent policy"""

#     def __init__(self, env, policy: callable):
#         assert isinstance(env.unwrapped, TronDuoEnv), "TronPlay wrapper requires a TronDuoEnv environment"
#         super().__init__(env)
#         self.policy = policy
#         self.env = env.unwrapped

#     def step(self, action : int):
#         assert action in [0, 1, 2, 3], f"Invalid action {action}."

#         self.heading1 = action

#         obs = TronDuoEnv.encode(self.env.state)[1]  # Get opponent's observation (the second one)
#         obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
#         opp_action = self.policy.act(obs)  # Get opponent action based on their observation
#         self.heading2 = (self.heading2 + (opp_action - 1)) % 4  # Because (left, forward, right)
        
#         dir1 = TronEnvBase.action_mapping[self.heading1]
#         dir2 = TronEnvBase.action_mapping[self.heading2]
    
#         result = self.tron.tick(dir1, dir2)
#         done = result != Result.PLAYING
#         obs = TronDuoEnv.encode(self.env.state)
#         reward = self.reward_dict[result]

#         if self.render:
#             self.view()

#         info = {"result": result} if done else {}
#         return obs, reward, done, False, info


class TorchObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.device = device

    def observation(self, obs):
        return torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)