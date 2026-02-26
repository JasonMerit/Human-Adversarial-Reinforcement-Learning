import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
import gymnasium as gym
import numpy as np

from . import utils

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
                
                # if event.key == self.pg.K_s:
                #     # Save state as state.npy
                #     np.save("state.npy", state)

        self.clock.tick(self.fps)
        
        return state, reward, done, _, info

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

def encode_observation(walls, bike_self, bike_other):
    occ = (walls > 0).astype(np.float32)

    you = np.zeros_like(occ)
    other = np.zeros_like(occ)

    x, y = bike_self
    you[y, x] = 1.0

    x, y = bike_other
    other[y, x] = 1.0

    return np.stack([occ, you, other], axis=0)

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
        obs_img = encode_observation(*obs)
        assert obs_img.shape == self.observation_space.shape, "Jason! Obs shape mismatch"
        return obs_img

class TronDualImage(gym.ObservationWrapper):
    """
    Transforms the dual state representation to an image ready for a CNN.
    """

    def __init__(self, env):
        super().__init__(env)
        tron = env.unwrapped.tron
        height, width = tron.height, tron.width
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=1, shape=(3, height, width), dtype=np.float32),
            gym.spaces.Box(low=0, high=1, shape=(3, height, width), dtype=np.float32)
        ))
    
    def observation(self, obs):
        obs1, obs2 = obs
        obs_img = encode_observation(*obs1), encode_observation(*obs2)
        assert obs_img[0].shape == self.observation_space.spaces[0].shape, f"Jason! Obs shape mismatch {obs_img[0].shape} vs {self.observation_space.spaces[0].shape}"
        assert obs_img[1].shape == self.observation_space.spaces[1].shape, f"Jason! Obs shape mismatch {obs_img[1].shape} vs {self.observation_space.spaces[1].shape}"
        return obs_img

class TronEgo(gym.Wrapper):
    """
    Assumes perspective of bike2 (agent)
    Transforms observation space to rotate view such that agent always heads upwards.
    Also reduces action space to [left, forward, right] relative to agent's perspective.

    orientation in [up, right, down, left]
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.orientation = 3  # First facing right and generally equal to absolute direction
        return self.observation(state), info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(utils.red(f"Invalid action!"))
        self.orientation = (self.orientation + (action - 1)) % 4  # Because (left, forward, right)
        state, reward, done, _, info = self.env.step(self.orientation)
        return self.observation(state), reward, done, _, info

    def observation(self, obs):
        return np.rot90(obs, k=self.orientation, axes=(1, 2)).copy()  # Copy to remove negative stride

class TronDualEgo(gym.Wrapper):
    """
    Transforms observation space to rotate view such that agent always heads upwards.
    Also reduces action space to [left, forward, right] relative to agent's perspective.

    orientation in [up, right, down, left]
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(3)))

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.heading1 = self.heading2 = 1  # First facing right and generally equal to absolute direction
        return self.observation(state), info

    def step(self, action):
        self.heading1 = (self.heading1 + (action[0] - 1)) % 4  # Because (left, forward, right)
        self.heading2 = (self.heading2 + (action[1] - 1)) % 4  # Because (left, forward, right)
        state, reward, done, _, info = self.env.step((self.heading1, self.heading2))
        return self.observation(state), reward, done, _, info

    def observation(self, obs):
        obs1, obs2 = obs
        obs1 = np.rot90(obs1, k=self.heading1, axes=(1, 2)).copy()  # Copy to remove negative stride
        obs2 = np.rot90(obs2, k=self.heading2, axes=(1, 2)).copy()  # Copy to remove negative stride
        return (obs1, obs2)