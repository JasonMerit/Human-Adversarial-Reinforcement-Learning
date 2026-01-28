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

background = (27, 79, 114)
white = (236, 240, 241)
green = (0,128,0)
darkGreen = (0,100,0)
red = (231, 76, 60)
darkRed = (241, 148, 138)
darkBlue = (40, 116, 166)

font = pg.font.SysFont("Agency FB", 65)

# Tron Bike Class
class Bike:
    def __init__(self, pos):
        self.x = pos[0]
        self.y = pos[1]

    def move(self, xdir, ydir):
        self.x += xdir
        self.y += ydir

    # Check if Bike Collides with Trail
    def is_hit(self, walls):
        if walls[self.y, self.x] != 0 or \
            self.x < 0 or self.x >= GRID_SIZE or \
                self.y < 0 or self.y >= GRID_SIZE:
            return True

class Tron:
    
    def reset(self):
        self.walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.heads = [[1, GRID_SIZE // 2], [GRID_SIZE - 2, GRID_SIZE // 2]]
        
        self.bike1 = Bike(self.heads[0])
        self.bike2 = Bike(self.heads[1])

        self.x1 = 1
        self.y1 = self.y2 = 0
        self.x2 = -1
    
    def move_bikes(self):
        self.walls[self.bike1.y, self.bike1.x] = 1
        self.walls[self.bike2.y, self.bike2.x] = 2
        self.bike1.move(self.x1, self.y1)
        self.bike2.move(self.x2, self.y2)
    
    def tick(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_q or event.key == pg.K_ESCAPE:
                    self.close()
                if event.key == pg.K_UP:
                    if not (self.x2 == 0 and self.y2 == 1):
                        self.x2 = 0
                        self.y2 = -1
                if event.key == pg.K_DOWN:
                    if not (self.x2 == 0 and self.y2 == -1):
                        self.x2 = 0
                        self.y2 = 1
                if event.key == pg.K_LEFT:
                    if not (self.x2 == 1 and self.y2 == 0):
                        self.x2 = -1
                        self.y2 = 0
                if event.key == pg.K_RIGHT:
                    if not (self.x2 == -1 and self.y2 == 0):
                        self.x2 = 1
                        self.y2 = 0
                if event.key == pg.K_w:
                    if not (self.x1 == 0 and self.y1 == 1):
                        self.x1 = 0
                        self.y1 = -1
                if event.key == pg.K_s:
                    if not (self.x1 == 0 and self.y1 == -1):
                        self.x1 = 0
                        self.y1 = 1
                if event.key == pg.K_a:
                    if not (self.x1 == 1 and self.y1 == 0):
                        self.x1 = -1
                        self.y1 = 0
                if event.key == pg.K_d:
                    if not (self.x1 == -1 and self.y1 == 0):
                        self.x1 = 1
                        self.y1 = 0

        self.move_bikes()
        self.heads = [[self.bike1.x, self.bike1.y], [self.bike2.x, self.bike2.y]]  # For drawing heads
        draw(self.walls, self.heads)

        bike1_hit = self.bike1.is_hit(self.walls)
        bike2_hit = self.bike2.is_hit(self.walls)

        if not (bike1_hit or bike2_hit):
            return
        if bike1_hit:
            if bike2_hit:
                self.gameOver(0)
            else:
                self.gameOver(1)
        else:
            self.gameOver(2)
                    
    def gameOver(self, number):
        if number == 0:
            text = font.render("Both the Players Collided!", True, white)
        else:
            text = font.render("Player %d Lost the Tron!" %(number), True, white)
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.close()
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_q or event.key == pg.K_ESCAPE:
                        self.close()
                    if event.key == pg.K_r:
                        self.reset()
                        return

            display.blit(text, (50, GRID_SIZE/2))
            
            pg.display.update()
            clock.tick(60)

    def close(self):
        pg.quit()
        sys.exit()

def draw(walls, heads):  # TODO Draw once and have bike in seperate layer
    # Draw checkered grid background
    logical_surface.fill(background)
    for x in range(0, GRID_SIZE):
        for y in range(0, GRID_SIZE):
            rect = pg.Rect(x, y, 1, 1)
            if walls[y, x] == 1:
                pg.draw.rect(logical_surface, red, rect)
            elif walls[y, x] == 2:
                pg.draw.rect(logical_surface, green, rect)
            elif (x + y) % 2 == 0:
                pg.draw.rect(logical_surface, (34, 49, 63), rect)
            else:
                pg.draw.rect(logical_surface, (41, 64, 82), rect)
    
    # Draw heads
    pg.draw.rect(logical_surface, darkRed, (heads[0][0], heads[0][1], 1, 1))
    pg.draw.rect(logical_surface, darkGreen, (heads[1][0], heads[1][1], 1, 1))
    scaled = pg.transform.scale(logical_surface, WINDOW_SIZE)
    display.blit(scaled, (0, 0))

    pg.display.update()
    clock.tick(FPS)

if __name__ == "__main__":
    tron = Tron()
    tron.reset()

    
    while True:
        tron.tick()
        # print(tron.walls[5], tron.walls[tron.bike1.y, tron.bike1.x])
        
        

