import sys
import pygame
import numpy as np

from tron_env.tron import Tron, Result

# --- Configuration ---
CELL_SIZE = 25                # pixels per cell

# --- Direction vectors ---
UP    = np.array([0, -1], dtype=np.int8)
DOWN  = np.array([0,  1], dtype=np.int8)
LEFT  = np.array([-1, 0], dtype=np.int8)
RIGHT = np.array([1,  0], dtype=np.int8)

DIR_MAP_P1 = {
    pygame.K_w: UP,
    pygame.K_s: DOWN,
    pygame.K_a: LEFT,
    pygame.K_d: RIGHT,
}

DIR_MAP_P2 = {
    pygame.K_UP: UP,
    pygame.K_DOWN: DOWN,
    pygame.K_LEFT: LEFT,
    pygame.K_RIGHT: RIGHT,
}

def draw_grid(screen, tron):
    screen.fill((0, 0, 0))
    for y in range(tron.height):
        for x in range(tron.width):
            cell = tron.walls[y, x]
            if cell == 1:
                color = (0, 255, 255)
            elif cell == 2:
                color = (255, 100, 0)
            else:
                if (x + y) % 2 == 0:
                    color = (40, 40, 40)
                else: color = (50, 50, 50)

            rect = pygame.Rect(
                x * CELL_SIZE,
                y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(screen, color, rect)

    pygame.display.flip()

def main(GRID_SIZE=40, TICK_RATE=10):
    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
    pygame.display.set_caption("Tron")

    clock = pygame.time.Clock()

    tron = Tron((GRID_SIZE, GRID_SIZE))
    tron.reset()

    dir1 = RIGHT.copy()
    dir2 = LEFT.copy()

    accumulator = 0.0
    dt = 1.0 / TICK_RATE

    while True:
        frame_time = clock.tick(240) / 1000.0  # high render FPS
        accumulator += frame_time

        # --- Input (event-driven, independent of tick rate) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

                if event.key in DIR_MAP_P1:
                    new_dir = DIR_MAP_P1[event.key]
                    if not np.array_equal(new_dir, -dir1):
                        dir1 = new_dir.copy()

                if event.key in DIR_MAP_P2:
                    new_dir = DIR_MAP_P2[event.key]
                    if not np.array_equal(new_dir, -dir2):
                        dir2 = new_dir.copy()

        # --- Fixed tick loop ---
        if accumulator >= dt:
            result = tron.tick(dir1, dir2)

            if result != Result.PLAYING:
                tron.reset()
                dir1 = RIGHT.copy()
                dir2 = LEFT.copy()

                if result == Result.BIKE1_CRASH:
                    print("JASON WON!")
                elif result == Result.BIKE2_CRASH:
                    print("KAROLINA WON!")
                else:
                    print("DRAW!")


            accumulator -= dt

        draw_grid(screen, tron)


if __name__ == "__main__":
    # Get GRID SIZE from args
    grid_size = int(sys.argv[1])
    tick_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(GRID_SIZE=grid_size, TICK_RATE=tick_rate)