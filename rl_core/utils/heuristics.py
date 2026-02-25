from collections import deque
import numpy as np

INF = float("inf")

DIRS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
]


def flood_fill(trails, start, dist):
    height, width = trails.shape
    queue = deque()

    y0, x0 = start
    dist[y0, x0] = 0
    queue.append((y0, x0))

    while queue:
        y, x = queue.popleft()

        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy

            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue

            if trails[ny, nx] != 0:
                continue

            if dist[ny, nx] != INF:
                continue

            dist[ny, nx] = dist[y, x] + 1
            queue.append((ny, nx))


def voronoi(trails, p1, p2):
    height, width = trails.shape

    dist1 = np.full((height, width), INF)
    dist2 = np.full((height, width), INF)

    flood_fill(trails, p1, dist1)
    flood_fill(trails, p2, dist2)

    region1 = 0
    region2 = 0

    for y in range(height):
        for x in range(width):

            if trails[y, x] != 0:
                continue

            d1 = dist1[y, x]
            d2 = dist2[y, x]

            if d1 == INF and d2 == INF:
                continue

            if d1 < d2:
                region1 += 1
            elif d2 < d1:
                region2 += 1
            # equal distance → ignored

    return region1 - region2