import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
import gymnasium as gym
import numpy as np
import time

from utils.helper import has_wrapper, bcolors

class TronView(gym.Wrapper):

    blue = (34, 49, 63)
    blue_alt = (41, 64, 82)
    green = (20, 180, 20)
    green_alt = (20, 220, 20)
    red = (180, 20, 20)
    red_alt = (220, 20, 20)

    def __init__(self, env, fps=10, scale=70):
        super().__init__(env)
        import pygame
        self.tron = self.env.unwrapped.tron
        
        self.pg = pygame
        self.pg.init()

        self.scale = scale
        width = self.tron.width
        height = self.tron.height
        self.window_size = (width * self.scale, height * self.scale)
        self.screen = self.pg.display.set_mode(self.window_size)
        self.trails_screen = self.pg.Surface((width, height), flags=self.pg.SRCALPHA)

        background = self.pg.Surface((width, height))
        for x in range(width):
            for y in range(height):
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
    
        self.prev1 = self.tron.bike1.pos.copy()
        self.prev2 = self.tron.bike2.pos.copy()
        return state, info
    
    def step(self, action):
        state, reward, done, _, info = self.env.step(action)
        assert self.env.observation_space.contains(state), f"Jason! Invalid state {state}"
        
        self.screen.blit(self.background, (0, 0))
        self.trails_screen.set_at((self.tron.bike1.pos[0], self.tron.bike1.pos[1]), self.green_alt)
        self.trails_screen.set_at((self.tron.bike2.pos[0], self.tron.bike2.pos[1]), self.red_alt)
        self.trails_screen.set_at((self.prev1[0], self.prev1[1]), self.green)
        self.trails_screen.set_at((self.prev2[0], self.prev2[1]), self.red)
        self._render()

        self.prev1 = self.tron.bike1.pos.copy()
        self.prev2 = self.tron.bike2.pos.copy()

        # Input
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                self.pg.quit()
                exit()
            if event.type == self.pg.KEYDOWN:
                if event.key == self.pg.K_q or event.key == self.pg.K_ESCAPE:
                    self.pg.quit()
                    exit()
                
                if event.key == self.pg.K_s:
                    # Save state as state.npy
                    np.save("state.npy", state)

        self.clock.tick(self.fps)
        
        return state, reward, done, _, info

    def _render(self):
        self.screen.blit(self.pg.transform.scale(self.trails_screen, self.window_size), (0, 0))
        self.pg.display.flip()
    
    @staticmethod
    def view_dual(state, scale):
        walls, bike1, bike2 = state[0]

        import pygame as pg
        pg.init()

        height, width = walls.shape
        window_size = (2 * width * scale + scale, height * scale)
        scale_size = (width * scale, height * scale)
        screen = pg.display.set_mode(window_size)
        pg.display.set_caption("Tron Game (State view)")

        you_surface = pg.Surface((width, height))
        for x in range(width):
            for y in range(height):
                if walls[y, x]:
                    color = TronView.green if walls[y, x] == 1 else TronView.red
                else:
                    color = TronView.blue if (x + y) % 2 else TronView.blue_alt
                you_surface.set_at((x, y), color)

        x, y = bike1
        you_surface.set_at((x, y), TronView.green_alt)
        x, y = bike2
        you_surface.set_at((x, y), TronView.red_alt)

        # ======= MIRROR =======
        walls, bike1, bike2 = state[1]
        opp_surface = pg.Surface((width, height))
        for x in range(width):
            for y in range(height):
                if walls[y, x]:
                    color = TronView.green if walls[y, x] == 2 else TronView.red
                else:
                    color = TronView.blue if (x + y) % 2 else TronView.blue_alt
                opp_surface.set_at((x, y), color)

        x, y = bike2
        opp_surface.set_at((x, y), TronView.green_alt)
        x, y = bike1
        opp_surface.set_at((x, y), TronView.red_alt)

        screen.blit(pg.transform.scale(you_surface, scale_size), (0, 0))
        screen.blit(pg.transform.scale(opp_surface, scale_size), (width * scale + scale, 0))
        pg.display.flip()

        TronView.wait()

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
                    if event.key in player1_keys:
                        p1_dir = player1_keys.index(event.key)
                    elif event.key in player2_keys:
                        p2_dir = player2_keys.index(event.key)
                
                clock.tick(30)

        return p1_dir, p2_dir

class TronImage(gym.ObservationWrapper):
    """
    Transforms the state representation to an image ready for a CNN.
    Used by TronEgo by default.
    """

    def __init__(self, env):
        super().__init__(env)
        tron = env.unwrapped.tron
        height, width = tron.height, tron.width
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, height, width), dtype=np.float32)
    
    def observation(self, obs):
        walls, bike1, bike2 = obs
        occ = (walls > 0).astype(np.float32)

        you = np.zeros_like(occ)
        other = np.zeros_like(occ)

        x, y = bike1
        you[y, x] = 1.0

        x, y = bike2
        other[y, x] = 1.0  

        # Stack into CNN input
        obs = np.stack([occ, you, other], axis=0)
        assert obs.shape == self.observation_space.shape, "Jason! Obs shape mismatch"
        return obs

class TronEgo(gym.Wrapper):
    """
    Transforms observation space to rotate view such that agent always heads upwards.
    Also reduces action space to [left, forward, right] relative to agent's perspective.

    orientation in [up, right, down, left]
    """

    def __init__(self, env):
        if not has_wrapper(env, TronImage):
            print(f"{bcolors.WARNING}TronEgo wrapper requires TronImage wrapper. Automatically adding TronImage wrapper.{bcolors.ENDC}")
            env = TronImage(env)
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.orientation = 1  # First facing right and generally equal to absolute direction
        return self.observation(state), info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"{bcolors.FAIL}Invalid action!{bcolors.ENDC}")
        self.orientation = (self.orientation + (action - 1)) % 4  # Because (left, forward, right)
        state, reward, done, _, info = self.env.step(self.orientation)
        return self.observation(state), reward, done, _, info

    def observation(self, obs):
        return np.rot90(obs, k=self.orientation, axes=(1, 2)).copy()  # Copy to remove negative stride

class TronTorch(gym.ObservationWrapper):
    """
    Transforms observation space to PyTorch tensor.
    """

    def __init__(self, env):
        import torch
        self.torch = torch
        if not has_wrapper(env, TronImage):
            print(f"{bcolors.WARNING}TronTorch wrapper requires TronImage wrapper. Automatically adding TronImage wrapper.{bcolors.ENDC}")
            env = TronImage(env)
        super().__init__(env)

    def observation(self, obs):
        return self.torch.tensor(obs, dtype=self.torch.float32).unsqueeze(0)