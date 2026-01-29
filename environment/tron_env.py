import numpy as np
import gymnasium as gym

from tron import Tron

class TronEnv(gym.Env):
    
    action_mapping = [(1,0), (-1,0), (0,1), (0,-1)]  # right, left, down, up

    def __init__(self, size=10, render=False):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict({
            'walls': gym.spaces.Box(low=0, high=2, shape=(size, size), dtype=np.int8),
            'bike1_pos': gym.spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int8),
            'bike2_pos': gym.spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int8),
        })


        self.tron = Tron(size)

        if render:
            import pygame
            self.pg = pygame
            self.pg.init()

            self.scale = 70
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
    
    def reset(self):
        self.tron.reset()
        self.dir1 = (1, 0)
        self.dir2 = (-1, 0)
        if hasattr(self, 'trails_screen'):
            self.trails_screen.fill((0, 0, 0, 0))
            self.prev1 = self.tron.bike1.pos
            self.prev2 = self.tron.bike2.pos
            self.render()
        return self._get_state()
    
    def step(self, action):
        self.dir1 = self.action_mapping[action]
        self.dir2 = self.action_mapping[1]
        # self.dir2 = self.action_mapping[np.random.randint(0, 4)]
    
        result = self.tron.tick(self.dir1, self.dir2)
        state = self._get_state()
        done = result != 0
        info = {'result': result}
        return state, done, info
    
    def _get_state(self):
        return (self.tron.walls.copy(), 
                tuple(self.tron.bike1.pos), 
                tuple(self.tron.bike2.pos))
    
    def render(self):
        assert hasattr(self, 'pg'), "Render not initialized. Set render=True in constructor."
        
        self.screen.blit(self.background, (0, 0))

        self.trails_screen.set_at((self.tron.bike1.pos[0], self.tron.bike1.pos[1]), (20, 220, 20))
        self.trails_screen.set_at((self.tron.bike2.pos[0], self.tron.bike2.pos[1]), (220, 20, 20))
        self.trails_screen.set_at((self.prev1[0], self.prev1[1]), (20, 180, 20))
        self.trails_screen.set_at((self.prev2[0], self.prev2[1]), (180, 20, 20))
        self.screen.blit(self.pg.transform.scale(self.trails_screen, self.window_size), (0, 0))
        self.pg.display.flip()

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

        self.clock.tick(self.fps)

if __name__ == "__main__":
    env = TronEnv(size=10, render=True)
    env.reset()
    
    done = False
    while True:
        # action = np.random.randint(0, 4)
        action = 0
        state, done, info = env.step(action)
        env.render()

        if done:
            env.reset()

