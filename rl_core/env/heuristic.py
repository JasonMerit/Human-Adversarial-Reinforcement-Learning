from collections import deque
import numpy as np

INF = float("inf")
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left

def get_best_action(state):
    """ Returns the action that maximizes voronoi score. """
    trails, you, other = state

    best_score = -np.inf
    best_action = None
    for action, dir in enumerate(DIRS):
        new_pos = you + dir

        if not(0 <= new_pos[0] < trails.shape[1]) or \
            not(0 <= new_pos[1] < trails.shape[0]) or \
            trails[new_pos[1], new_pos[0]] != 0:
            continue

        score = voronoi(trails, new_pos, other)

        if score > best_score:
            best_score = score
            best_action = action

    if best_action is None:
        return 0  # If no valid moves, just pick up (or any default)
    return best_action

def flood_fill(trails, start, dist):
    height, width = trails.shape
    queue = deque()

    x0, y0 = start
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

def get_territories(trails, p1, p2):
    height, width = trails.shape

    dist1 = np.full((height, width), INF)
    dist2 = np.full((height, width), INF)
    territories = np.zeros((height, width), dtype=np.int8)

    flood_fill(trails, p1, dist1)
    flood_fill(trails, p2, dist2)

    for y in range(height):
        for x in range(width):

            if trails[y, x] != 0:
                continue

            d1 = dist1[y, x]
            d2 = dist2[y, x]

            if d1 == INF and d2 == INF:
                continue

            if d1 < d2:
                territories[y, x] = 1
            elif d2 < d1:
                territories[y, x] = 2
            else:
                territories[y, x] = 3  # Battlefront
            
    return territories

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