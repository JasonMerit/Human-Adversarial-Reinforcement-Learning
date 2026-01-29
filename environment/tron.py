import numpy as np

class Bike:
    
    def __init__(self, pos):
        self.pos = pos

    def move(self, dx, dy):
        self.pos[0] += dx
        self.pos[1] += dy

    def is_hit(self, walls):
        x, y = self.pos
        if y < 0 or y >= len(walls) or \
            x < 0 or x >= len(walls[0]) or \
            walls[y, x] != 0:
            return True

class Tron:
    
    def __init__(self, grid_size=10):
        self.grid_size = grid_size

    def reset(self):
        self.walls = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.bike1 = Bike([1, self.grid_size // 2])
        self.bike2 = Bike([self.grid_size - 2, self.grid_size // 2])

    def tick(self, dir1, dir2):        
        self._move_bikes(dir1, dir2)
        return self._check_collisions()
    
    def _move_bikes(self, dir1, dir2):
        self.walls[self.bike1.pos[1], self.bike1.pos[0]] = 1
        self.walls[self.bike2.pos[1], self.bike2.pos[0]] = 2
        self.bike1.move(dir1[0], dir1[1])
        self.bike2.move(dir2[0], dir2[1])
    
    def _check_collisions(self):
        bike1_hit = self.bike1.is_hit(self.walls)
        bike2_hit = self.bike2.is_hit(self.walls)

        if (bike1_hit and bike2_hit) or (self.bike1.pos == self.bike2.pos):
            return 3  
        if bike1_hit:
            return 1  
        if bike2_hit:
            return 2  
        return 0
        
if __name__ == "__main__":                 
    import pygame as pg
    import sys
    
    SCALE = 70
    GRID_SIZE = 10

    WINDOW_SIZE = (GRID_SIZE * SCALE, GRID_SIZE * SCALE)
    pg.init()
    display = pg.display.set_mode(WINDOW_SIZE)
    pg.display.set_caption("Tron Game")
    logical_surface = pg.Surface((GRID_SIZE, GRID_SIZE))

    FPS = 4
    clock = pg.time.Clock()

    white = (236, 240, 241)
    green = (0,128,0)
    darkGreen = (0,100,0)
    red = (231, 76, 60)
    darkRed = (241, 148, 138)
    blue = (41, 64, 82)
    blue_alt = (34, 49, 63)

    font = pg.font.SysFont("Agency FB", 65)

    def close():        
        pg.quit()
        sys.exit()

    def gameOver(number):
        if number == 3:
            text = font.render("Both Actors Collided!", True, white)
        else:
            text = font.render("Actor %d Lost!" %(number), True, white)
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    close()
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_q or event.key == pg.K_ESCAPE:
                        close()
                    if event.key == pg.K_r:
                        tron.reset()
                        return

            display.blit(text, (50, GRID_SIZE/2))
            
            pg.display.update()
            clock.tick(60)

    def draw(walls, bike1_pos, bike2_pos):  # TODO Draw once and have bike in seperate layer
        for x in range(0, GRID_SIZE):
            for y in range(0, GRID_SIZE):
                if walls[y, x] == 1:
                    logical_surface.set_at((x, y), red)
                elif walls[y, x] == 2:
                    logical_surface.set_at((x, y), green)
                elif (x + y) % 2 == 0:
                    logical_surface.set_at((x, y), blue)
                else:
                    logical_surface.set_at((x, y), blue_alt)
        
        # Draw heads
        logical_surface.set_at((bike1_pos[0], bike1_pos[1]), darkRed)
        logical_surface.set_at((bike2_pos[0], bike2_pos[1]), darkGreen)
        
        scaled = pg.transform.scale(logical_surface, WINDOW_SIZE)
        display.blit(scaled, (0, 0))

        pg.display.update()
        clock.tick(FPS)

    def get_inputs(dir1, dir2):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                close()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_q or event.key == pg.K_ESCAPE:
                    close()

                if event.key == pg.K_UP:
                    if not (dir2 == (0, 1)):
                        dir2 = (0, -1)
                if event.key == pg.K_DOWN:
                    if not (dir2 == (0, -1)):
                        dir2 = (0, 1)
                if event.key == pg.K_LEFT:
                    if not (dir2 == (1, 0)):
                        dir2 = (-1, 0)
                if event.key == pg.K_RIGHT:
                    if not (dir2 == (-1, 0)):
                        dir2 = (1, 0)

                if event.key == pg.K_w:
                    if not (dir1 == (0, 1)):
                        dir1 = (0, -1)
                if event.key == pg.K_s:
                    if not (dir1 == (0, -1)):
                        dir1 = (0, 1)
                if event.key == pg.K_a:
                    if not (dir1 == (1, 0)):
                        dir1 = (-1, 0)
                if event.key == pg.K_d:
                    if not (dir1 == (-1, 0)):
                        dir1 = (1, 0)

        return dir1, dir2


    tron = Tron(GRID_SIZE)
    tron.reset()
    dir1 = (1, 0)
    dir2 = (-1, 0)

    while True:
        dir1, dir2 = get_inputs(dir1, dir2)
        result = tron.tick(dir1, dir2)
        
        state = (tron.walls, tron.bike1.pos, tron.bike2.pos)
        draw(*state)
        
        if result != 0:
            gameOver(result)
            tron.reset()
            dir1 = (1, 0)
            dir2 = (-1, 0)
        
        

