import numpy as np
import gymnasium as gym

from environment.tron import Tron

class TronEnv(gym.Env):
    
    action_mapping = [(1,0), (-1,0), (0,1), (0,-1)]  # right, left, down, up
    reward_mapping = [0.0, -1.0, 1, 0.5]  # playing, lose, win, draw

    def __init__(self, size=10):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(size, size, 3), dtype=float)

        self.tron = Tron(size)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tron.reset()
        self.dir1 = (1, 0)
        self.dir2 = (-1, 0)
            
        return self._get_state(), {}
    
    def step(self, action):
        assert self.action_space.contains(action), "Jason! Invalid Action"
        self.dir1 = self.action_mapping[action]
        self.dir2 = self.action_mapping[1]
        # self.dir2 = self.action_mapping[np.random.randint(0, 4)]
    
        result = self.tron.tick(self.dir1, self.dir2)
        state = self._get_state()
        reward = self.reward_mapping[result]
        done = result != 0
        info = {'result': result}
        return state, reward, done, False, info
    
    def _get_state(self):
        walls = self.tron.walls.copy()
        occ = (walls > 0).astype(float)

        bike1 = np.zeros_like(occ)
        x1, y1 = self.tron.bike1.pos
        bike1[y1, x1] = 1.0

        bike2 = np.zeros_like(occ)
        x2, y2 = self.tron.bike2.pos
        bike2[y2, x2] = 1.0

        # Stack into CNN input
        state = np.stack([occ, bike1, bike2], axis=-1)
        assert state.shape == self.observation_space.shape, "Jason! State shape mismatch"
        return state
    
    
class TronView(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        import pygame
        
        self.pg = pygame
        self.pg.init()

        self.scale = 70
        size = env.tron.size
        self.window_size = (size * self.scale, size * self.scale)
        self.screen = self.pg.display.set_mode(self.window_size)
        self.trails_screen = self.pg.Surface((size, size), flags=self.pg.SRCALPHA)

        background = self.pg.Surface((size, size))
        for x in range(size):
            for y in range(size):
                color = (34, 49, 63) if (x + y) % 2 else (41, 64, 82)
                background.set_at((x, y), color)
        self.background = self.pg.transform.scale(background, self.window_size)

        self.pg.display.set_caption("Tron Game")
        
        self.clock = self.pg.time.Clock()
        self.fps = 4

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.screen.blit(self.background, (0, 0))
        self.trails_screen.fill((0, 0, 0, 0))  # Clear trails with transparency
        self._render()
    
        self.prev1 = self.env.tron.bike1.pos.copy()
        self.prev2 = self.env.tron.bike2.pos.copy()

        return state, info
    
    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        
        self.screen.blit(self.background, (0, 0))

        self.trails_screen.set_at((self.env.tron.bike1.pos[0], self.env.tron.bike1.pos[1]), (20, 220, 20))
        self.trails_screen.set_at((self.env.tron.bike2.pos[0], self.env.tron.bike2.pos[1]), (220, 20, 20))
        self.trails_screen.set_at((self.prev1[0], self.prev1[1]), (20, 180, 20))
        self.trails_screen.set_at((self.prev2[0], self.prev2[1]), (180, 20, 20))
        self._render()

        self.prev1 = self.env.tron.bike1.pos.copy()
        self.prev2 = self.env.tron.bike2.pos.copy()

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
        
        return state, reward, done, truncated, info

    def _render(self):
        self.screen.blit(self.pg.transform.scale(self.trails_screen, self.window_size), (0, 0))
        self.pg.display.flip()

