import numpy as np

from .constants import Color

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def has_wrapper(env, wrapper_class):
    """Check if `env` or any of its wrappers is an instance of `wrapper_class`."""
    current = env
    while current:
        if isinstance(current, wrapper_class):
            return True
        # Stop if there is no further wrapped env
        if not hasattr(current, 'env'):
            break
        current = current.env
    return False


class StateViewer:

    def __init__(self, size, scale=70, fps=10):
        self.scale = scale
        self.fps = fps
        import pygame
        self.pg = pygame
        self.pg.init()

        self.width, self.height = size
        self.window_size = (self.width * scale, self.height * scale)
        self.screen = self.pg.display.set_mode(self.window_size)
        self.pg.display.set_caption("Tron Game (State view)")
        self.surface = self.pg.Surface(size)

        self.clock = self.pg.time.Clock()
    
    def view(self, state):
        walls, bike1, bike2 = state

        self.draw_walls(walls)

        # Heads
        self.surface.set_at(bike1, Color.GREEN_ALT)
        self.surface.set_at(bike2, Color.RED_ALT)
        self.screen.blit(self.pg.transform.scale(self.surface, self.window_size), (0, 0))
        self.pg.display.flip()

        self.wait()
    
    def view_image(self, image):
        image = image.squeeze(0)
        walls, bike1, bike2 = image
        
        self.draw_walls(walls)

        # Heads
        y, x = np.argwhere(bike1 == 1)
        self.surface.set_at((x, y), Color.GREEN_ALT)
        y, x = np.argwhere(bike2 == 1)
        self.surface.set_at((x, y), Color.RED_ALT)

        self.screen.blit(self.pg.transform.scale(self.surface, self.window_size), (0, 0))
        self.pg.display.flip()

        self.wait()

    def draw_walls(self, walls):
        for x in range(self.width):
            for y in range(self.height):
                if walls[y, x]:
                    color = Color.GREEN if walls[y, x] == 1 else Color.RED
                else:
                    color = Color.BLUE if (x + y) % 2 else Color.BLUE_ALT
                self.surface.set_at((x, y), color)
    
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
