from pyclbr import Class
import pygame as pg
import sys
import numpy as np

GRID_SIZE = 10
SCALE = 70

LOGICAL_SIZE = (GRID_SIZE, GRID_SIZE)
WINDOW_SIZE = (GRID_SIZE * SCALE, GRID_SIZE * SCALE)

pg.init()
display = pg.display.set_mode(WINDOW_SIZE)
pg.display.set_caption("Tron Game")
logical_surface = pg.Surface(LOGICAL_SIZE)

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

# Tron Bike Class
class Bike:
    def __init__(self, pos):
        self.pos = pos

    def move(self, dx, dy):
        self.pos[0] += dx
        self.pos[1] += dy

    # Check if Bike Collides with Trail
    def is_hit(self, walls):
        print(self.pos)
        if self.pos[1] < 0 or self.pos[1] >= GRID_SIZE or \
            self.pos[0] < 0 or self.pos[0] >= GRID_SIZE or \
            walls[self.pos[1], self.pos[0]] != 0:
            return True

class Tron:
    
    def reset(self):
        self.walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.bike1 = Bike([1, GRID_SIZE // 2])
        self.bike2 = Bike([GRID_SIZE - 2, GRID_SIZE // 2])

    def move_bikes(self, dir1, dir2):
        self.walls[self.bike1.pos[1], self.bike1.pos[0]] = 1
        self.walls[self.bike2.pos[1], self.bike2.pos[0]] = 2
        self.bike1.move(dir1[0], dir1[1])
        self.bike2.move(dir2[0], dir2[1])
    
    def tick(self, dir1, dir2):        
        self.move_bikes(dir1, dir2)

        bike1_hit = self.bike1.is_hit(self.walls)
        bike2_hit = self.bike2.is_hit(self.walls)

        if not (bike1_hit or bike2_hit):
            return 0
        if bike1_hit:
            if bike2_hit:
                return 3
            else:
                return 1
        else:
            return 2
                    
    def close(self):        
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
                tron.close()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_q or event.key == pg.K_ESCAPE:
                    tron.close()
                if event.key == pg.K_r:
                    tron.reset()
                    return

        display.blit(text, (50, GRID_SIZE/2))
        
        pg.display.update()
        clock.tick(60)


def draw(walls, bike1_pos, bike2_pos):  # TODO Draw once and have bike in seperate layer
    # Draw checkered grid background
    for x in range(0, GRID_SIZE):
        for y in range(0, GRID_SIZE):
            rect = pg.Rect(x, y, 1, 1)
            if walls[y, x] == 1:
                pg.draw.rect(logical_surface, red, rect)
            elif walls[y, x] == 2:
                pg.draw.rect(logical_surface, green, rect)
            elif (x + y) % 2 == 0:
                pg.draw.rect(logical_surface, blue, rect)
            else:
                pg.draw.rect(logical_surface, blue_alt, rect)
    
    # Draw heads
    pg.draw.rect(logical_surface, darkRed, (bike1_pos[0], bike1_pos[1], 1, 1))
    pg.draw.rect(logical_surface, darkGreen, (bike2_pos[0], bike2_pos[1], 1, 1))
    scaled = pg.transform.scale(logical_surface, WINDOW_SIZE)
    display.blit(scaled, (0, 0))

    pg.display.update()
    clock.tick(FPS)

def get_inputs(dir1, dir2):
    for event in pg.event.get():
        if event.type == pg.QUIT:
            tron.close()
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_q or event.key == pg.K_ESCAPE:
                tron.close()

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



if __name__ == "__main__":
    tron = Tron()
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
        
        

