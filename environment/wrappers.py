import gymnasium as gym
import numpy as np

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
        
        self.pg = pygame
        self.pg.init()

        self.scale = scale
        width = env.unwrapped.tron.width
        height = env.unwrapped.tron.height
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

    @staticmethod
    def view(state, scale):
        walls, bike1, bike2 = state

        import pygame as pg
        pg.init()

        _, height, width = state.shape
        window_size = (width * scale, height * scale)
        screen = pg.display.set_mode(window_size)
        pg.display.set_caption("Tron Game (State view)")

        surface = pg.Surface((width, height))
        for x in range(width):
            for y in range(height):
                if walls[y, x]:
                    color = (180, 180, 20)
                else:
                    color = TronView.blue if (x + y) % 2 else TronView.blue_alt
                surface.set_at((x, y), color)

        # Heads
        try:
            y, x = np.argwhere(bike1 == 1)[0]  # Flipped coordinates
            surface.set_at((x, y), TronView.green_alt)
        except IndexError:
            pass
        try:
            y, x = np.argwhere(bike2 == 1)[0]  # Flipped coordinates
            surface.set_at((x, y), TronView.red_alt)
        except IndexError:
            pass

        screen.blit(pg.transform.scale(surface, window_size), (0, 0))
        pg.display.flip()
        
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
        

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)

        self.screen.blit(self.background, (0, 0))
        self.trails_screen.fill((0, 0, 0, 0))  # Clear trails with transparency

        # Fill in walls (neccesary after first move is made before env starts)
        walls = self.env.unwrapped.tron.walls
        for y, row in enumerate(walls):
            for x, cell in enumerate(row):
                if cell == 0: continue
                color = self.green if cell == 1 else self.red
                self.trails_screen.set_at((x, y), color)
                
        self._render()
    
        self.prev1 = self.env.unwrapped.tron.bike1.pos.copy()
        self.prev2 = self.env.unwrapped.tron.bike2.pos.copy()
        return state, info
    
    def step(self, action):
        state, reward, done, _, info = self.env.step(action)
        
        self.screen.blit(self.background, (0, 0))
        self.trails_screen.set_at((self.env.unwrapped.tron.bike1.pos[0], self.env.unwrapped.tron.bike1.pos[1]), self.green_alt)
        self.trails_screen.set_at((self.env.unwrapped.tron.bike2.pos[0], self.env.unwrapped.tron.bike2.pos[1]), self.red_alt)
        self.trails_screen.set_at((self.prev1[0], self.prev1[1]), self.green)
        self.trails_screen.set_at((self.prev2[0], self.prev2[1]), self.red)
        self._render()

        self.prev1 = self.env.unwrapped.tron.bike1.pos.copy()
        self.prev2 = self.env.unwrapped.tron.bike2.pos.copy()

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


class TronEgo(gym.Wrapper):
    """
    Transforms observation space to rotate view such that agent always heads upwards.
    Also reduces action space to [left, forward, right] relative to agent's perspective.

    orientation in [up, right, down, left]
    """

    def __init__(self, env):
        super().__init__(env)
        self.original_action_space = env.action_space
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.orientation = 1  # First facing right and generally equal to absolute direction
        return self.observation(state), info

    def step(self, action):
        assert self.action_space.contains(action), "Jason! Invalid action"
        self.orientation = (self.orientation + (action - 1)) % 4  # Because (left, forward, right)
        state, reward, done, _, info = self.env.step(self.orientation)
        return self.observation(state), reward, done, _, info

    def observation(self, obs):
        return np.rot90(obs, k=self.orientation, axes=(1, 2)).copy()  # Copy to remove negative stride
