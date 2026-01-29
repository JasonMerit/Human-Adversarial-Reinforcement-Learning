import numpy as np
import gymnasium as gym

from environment.tron import Bike, Tron

class TronEnv(gym.Env):
    
    action_mapping = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
    reward_mapping = [0.0, -1.0, 1, 0.5]  # playing, lose, win, draw

    def __init__(self, size=10):
        self.tron = Tron(size)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(size, size, 3), dtype=float)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tron.reset()
        self.last_opponent_action = 0
        # self.last_opponent_action = 3  # Start moving left
            
        return self._get_state(), {'result': 0}
    
    def step(self, action):
        assert self.action_space.contains(action), "Jason! Invalid Action"
        dir1 = self.action_mapping[action]
        dir2 = self._get_opponent_action()
    
        result = self.tron.tick(dir1, dir2)
        done = result != 0
        state = self._get_state(done)
        reward = self.reward_mapping[result]
        info = {'result': result}
        return state, reward, done, False, info
    
    def _get_state(self, done=False):
        walls = self.tron.walls.copy()
        occ = (walls > 0).astype(float)

        bike1 = np.zeros_like(occ)
        bike2 = np.zeros_like(occ)  
        if not done:
            x1, y1 = self.tron.bike1.pos
            bike1[y1, x1] = 1.0

            x2, y2 = self.tron.bike2.pos
            bike2[y2, x2] = 1.0  # Out of bounds - Skip if done.

        # Stack into CNN input
        state = np.stack([occ, bike1, bike2], axis=-1)
        assert state.shape == self.observation_space.shape, "Jason! State shape mismatch"
        return state
    
    def _get_opponent_action(self):
        """
        Semi-deterministic agent that repeats its last action unless that action is invalid.
        In that case, it randomly selects a valid action.

        :return: Direction tuple (dx, dy) for opponent bike
        """
        action = self.last_opponent_action
        possible_actions = {0, 1, 2, 3} - {action, (action + 2) % 4}

        while not self._is_valid_action_bike2(action) and len(possible_actions) > 0:
            action = self.np_random.choice(list(possible_actions))
            possible_actions.remove(action)

        self.last_opponent_action = action
        return self.action_mapping[action]
    
    def _is_valid_action_bike2(self, action):
        dir = self.action_mapping[action]
        pos = self.tron.bike2.pos
        new_pos = [pos[0] + dir[0], pos[1] + dir[1]]
        return not Bike.is_hit(new_pos, self.tron.walls)
    
    
class TronView(gym.Wrapper):
    
    def __init__(self, env, fps=4):
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
        self.fps = fps

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        self.screen.blit(self.background, (0, 0))
        self.trails_screen.fill((0, 0, 0, 0))  # Clear trails with transparency
        self._render()
    
        self.prev1 = self.env.tron.bike1.pos.copy()
        self.prev2 = self.env.tron.bike2.pos.copy()

        return state, info
    
    def step(self, action):
        state, reward, done, _, info = self.env.step(action)
        
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
        
        return state, reward, done, _, info

    def _render(self):
        self.screen.blit(self.pg.transform.scale(self.trails_screen, self.window_size), (0, 0))
        self.pg.display.flip()

