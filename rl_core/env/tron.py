import numpy as np

class Result:
    DRAW = 0
    BIKE2_CRASH = 1
    BIKE1_CRASH = 2
    PLAYING = -1

class Tron:
    
    def __init__(self, size):
        self.size = size

    def reset(self):
        self.walls = np.zeros((self.size, self.size), dtype=np.int8)
        self.pos1 = np.array([self.size // 6, self.size // 2], dtype=np.int8)
        self.pos2 = np.array([5 * self.size // 6, self.size // 2], dtype=np.int8)
        # self.walls[self.pos1[1], self.pos1[0]] = 1
        # self.walls[self.pos2[1], self.pos2[0]] = 2

    def tick(self, dir1, dir2):
        """
        Updates the game state by moving the bikes and checking for collisions.
        
        :param dir1: Direction for bike1 as a tuple (dx, dy)
        :param dir2: Direction for bike2 as a tuple (dx, dy)
        :return: An integer indicating the result of the tick:
                 -1 - No collision
                 2 - Bike1 collided
                 1 - Bike2 collided
                 0 - Both bikes collided (draw)
        """
        assert type(dir1) == np.ndarray and type(dir2) == np.ndarray, f"Invalid direction type : {type(dir1)}, {type(dir2)}"
        assert dir1.shape == (2, ) and dir2.shape == (2, ), f"Invalid direction shape : {dir1.shape}, {dir2.shape}"
        
        # Mark previous cells
        self.walls[self.pos1[1], self.pos1[0]] = 1
        self.walls[self.pos2[1], self.pos2[0]] = 2

        # Move bikes
        self.pos1 = self.pos1 + dir1
        self.pos2 = self.pos2 + dir2
        self.pos1 = np.clip(self.pos1, 0, self.size - 1)
        self.pos2 = np.clip(self.pos2, 0, self.size - 1)

        # Detect crashes
        crash1 = self.walls[self.pos1[1], self.pos1[0]] != 0
        crash2 = self.walls[self.pos2[1], self.pos2[0]] != 0

        if (crash1 and crash2) or all(self.pos1 == self.pos2):
            return Result.DRAW  
        if crash1:
            return Result.BIKE1_CRASH  
        if crash2:
            return Result.BIKE2_CRASH  

        return Result.PLAYING
        
