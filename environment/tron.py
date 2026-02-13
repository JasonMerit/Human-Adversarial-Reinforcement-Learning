import numpy as np
from utils.helper import bcolors

class Bike:
    
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=np.int8)

    def move(self, vel):
        self.pos += vel

    def is_hit(self, walls):
        x, y = self.pos
        return not 0 <= y < len(walls) or not 0 <= x < len(walls[0]) or walls[y, x] != 0

class Tron:
    
    def __init__(self, width, height):
        self.width = width
        self.height = height


    def reset(self):
        self.walls = np.zeros((self.height, self.width), dtype=np.int8)
        self.bike1 = Bike([1, self.height // 2])
        self.bike2 = Bike([self.width - 2, self.height // 2])

    def tick(self, dir1, dir2):
        """
        Updates the game state by moving the bikes and checking for collisions.
        
        :param dir1: Direction for bike1 as a tuple (dx, dy)
        :param dir2: Direction for bike2 as a tuple (dx, dy)
        :return: An integer indicating the result of the tick:
                 0 - No collision
                 1 - Bike1 collided
                 2 - Bike2 collided
                 3 - Both bikes collided (draw)
        """
        assert type(dir1) == np.ndarray and type(dir2) == np.ndarray, f"{bcolors.FAIL}Invalid direction type : {type(dir1)}, {type(dir2)}{bcolors.ENDC}"
        assert dir1.shape == (2, ) and dir2.shape == (2, ), f"{bcolors.FAIL}Invalid direction shape : {dir1.shape}, {dir2.shape}{bcolors.ENDC}"
        self._move_bikes(dir1, dir2)
        return self._check_collisions()
    
    def _move_bikes(self, dir1, dir2):
        self.walls[self.bike1.pos[1], self.bike1.pos[0]] = 1
        self.walls[self.bike2.pos[1], self.bike2.pos[0]] = 2
        self.bike1.move(dir1)
        self.bike2.move(dir2)

    def _check_collisions(self):
        bike1_hit = self.bike1.is_hit(self.walls)
        bike2_hit = self.bike2.is_hit(self.walls)

        if (bike1_hit and bike2_hit) or all(self.bike1.pos == self.bike2.pos):
            return 3  
        if bike1_hit:
            return 1  
        if bike2_hit:
            return 2  
        return 0
        
