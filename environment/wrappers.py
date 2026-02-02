import gymnasium as gym

class TronView(gym.Wrapper):
    
    def __init__(self, env, fps, scale):
        super().__init__(env)
        import pygame
        
        self.pg = pygame
        self.pg.init()

        self.scale = scale
        width = env.tron.width
        height = env.tron.height
        self.window_size = (width * self.scale, height * self.scale)
        self.screen = self.pg.display.set_mode(self.window_size)
        self.trails_screen = self.pg.Surface((width, height), flags=self.pg.SRCALPHA)

        background = self.pg.Surface((width, height))
        for x in range(width):
            for y in range(height):
                color = (34, 49, 63) if (x + y) % 2 else (41, 64, 82)
                background.set_at((x, y), color)
        self.background = self.pg.transform.scale(background, self.window_size)

        self.pg.display.set_caption("Tron Game")
        
        self.clock = self.pg.time.Clock()
        self.fps = fps

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)

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

