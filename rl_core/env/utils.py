import numpy as np

class Color:
    BLUE = (34, 49, 63)
    BLUE_ALT = (41, 64, 82)
    GREEN = (20, 180, 20)
    GREEN_ALT = (20, 220, 20)
    RED = (180, 20, 20)
    RED_ALT = (220, 20, 20)
    GREY = (128, 128, 128)

def red(text) -> str:
    return '\033[91m' + str(text) + '\033[0m'

def green(text) -> str:
    return '\033[92m' + str(text) + '\033[0m'

def yellow(text) -> str:
    return '\033[93m' + str(text) + '\033[0m'

def blue(text) -> str:
    return '\033[94m' + str(text) + '\033[0m'

def cyan(text) -> str:
    return '\033[96m' + str(text) + '\033[0m'

class StateViewer:
    """Designed to be ignorant of tron, draw exactly whats provided in the state"""

    def __init__(self, size, scale=70, fps=10, single=True):
        self.scale = scale
        self.fps = fps
        import pygame
        self.pg = pygame
        self.pg.init()

        self.size = size
        self.window_size = (size * scale, size * scale) if single else (2 * size * scale + scale, size * scale)
        self.screen = self.pg.display.set_mode(self.window_size)
        self.pg.display.set_caption("Tron Game (State view)")
        self.surface = self.pg.Surface((size, size))
        self.transparent_surface = self.pg.Surface((size, size), self.pg.SRCALPHA)

        self.clock = self.pg.time.Clock()
    
    # def view(self, state):
    #     walls, bike1, bike2 = state

    #     self.draw_walls_to_surface(walls, self.surface)

    #     # Heads
    #     self.surface.set_at(bike1, Color.GREEN_ALT)
    #     self.surface.set_at(bike2, Color.RED_ALT)
    #     self.screen.blit(self.pg.transform.scale(self.surface, self.window_size), (0, 0))
    #     self.pg.display.flip()

    #     self.wait()
    
    def view(self, state):
        walls, bike1, bike2 = state

        self.draw_walls_to_surface(walls, self.surface)

        # Heads
        y, x = np.argwhere(bike1 == 1)[0]
        self.surface.set_at((x, y), Color.GREEN_ALT)
        y, x = np.argwhere(bike2 == 1)[0]
        self.surface.set_at((x, y), Color.RED_ALT)

        self.screen.blit(self.pg.transform.scale(self.surface, self.window_size), (0, 0))
        self.pg.display.flip()

        self.wait()

    def get_dual_action(self):
        action1 = None
        action2 = None

        keymap_p1 = {self.pg.K_a: 0, self.pg.K_w: 1, self.pg.K_d: 2}
        keymap_p2 = {self.pg.K_LEFT: 0, self.pg.K_UP: 1, self.pg.K_RIGHT: 2}

        while action1 is None or action2 is None:
            for event in self.pg.event.get():
                if event.type == self.pg.QUIT:
                    self.pg.quit()
                    exit()

                if event.type == self.pg.KEYDOWN:
                    if event.key in (self.pg.K_q, self.pg.K_ESCAPE):
                        self.pg.quit()
                        exit()
                    if event.key in keymap_p1:
                        action1 = keymap_p1[event.key]
                    if event.key in keymap_p2:
                        action2 = keymap_p2[event.key]

        return action1, action2
    
    def view_dual(self, images):
        scale_size = (self.size * self.scale, self.size * self.scale)

        walls, bike1, bike2 = images[0]

        self.draw_walls_to_surface(walls, self.surface)

        y, x = np.argwhere(bike1 == 1)[0]
        self.surface.set_at((x, y), Color.GREEN_ALT)
        y, x = np.argwhere(bike2 == 1)[0]
        self.surface.set_at((x, y), Color.RED_ALT)

        # ======= MIRROR =======
        walls, bike1, bike2 = images[1]
        
        opp_surface = self.pg.Surface((self.size, self.size))
        self.draw_walls_to_surface(walls, opp_surface)
        y, x = np.argwhere(bike1 == 1)[0]
        opp_surface.set_at((x, y), Color.RED_ALT)
        y, x = np.argwhere(bike2 == 1)[0]
        opp_surface.set_at((x, y), Color.GREEN_ALT)

        self.screen.blit(self.pg.transform.scale(self.surface, scale_size), (0, 0))
        self.screen.blit(self.pg.transform.scale(opp_surface, scale_size), (self.size * self.scale + self.scale, 0))
        self.pg.display.flip()

        self.wait()

    def draw_walls_to_surface(self, walls, surface):
        for x in range(self.size):
            for y in range(self.size):
                if walls[y, x]:
                    color = Color.GREY
                    # color = Color.GREEN if walls[y, x] == 1 else Color.RED
                else:
                    color = Color.BLUE if (x + y) % 2 else Color.BLUE_ALT
                surface.set_at((x, y), color)
    
    def wait(self):
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                self.pg.quit()
                exit()
            if event.type == self.pg.KEYDOWN:
                if event.key == self.pg.K_q or event.key == self.pg.K_ESCAPE:
                    self.pg.quit()
                    exit()

        self.clock.tick(self.fps)