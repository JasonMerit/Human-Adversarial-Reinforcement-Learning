import numpy as np
import timeit
from collections import deque

FRIENDLY = 1
OPPONENT = 2
ARTICULATION = 3
SIZE = 10, 10

def id_mat(r,c):
    return r*SIZE[1] + c

def get_state(walls, player, opponent):
    state = walls.copy()
    x, y = player.pos
    state[y, x] = FRIENDLY

    x, y = opponent.pos
    state[y, x] = OPPONENT
    return state

def grid_neighbors(row,col):
    maxrow = SIZE[0]
    maxcol = SIZE[1]
    l = []
    if (row+1 < maxrow):
        l += [(row+1, col)]
    if (row > 0):
        l += [(row-1, col)]
    if (col+1 < maxcol):
        l += [(row, col+1)]
    if (col > 0):
        l += [(row, col-1)]
    return l

def dijkstra(state, head):
    hc, hr = head
    dists = np.full(SIZE, np.inf)
    
    # Initialize BFS queue with (row, col, dist)
    q = deque()
    q.append((hr, hc, 0))
    dists[hr, hc] = 0

    while q:
        cr, cc, dist = q.popleft()
        dists[cr, cc] = dist

        for nr, nc in grid_neighbors(cr, cc):
            if state[nr, nc] != 0 or dists[nr, nc] < np.inf:
                continue
            q.append((nr, nc, dist + 1))

    return dists

def hopcroft_tarjan(state):
    parents = np.zeros(SIZE)
    parents[:] = np.inf
    parents[0,0] = -1
    visited = np.zeros(SIZE)
    low = np.zeros(SIZE)
    low[:] = np.inf
    depths = np.zeros(SIZE)
    depths[:] = np.inf
    rec_hopcroft_tarjan(state, 0, 0, 0, depths, parents, visited, low)

def rec_hopcroft_tarjan(state, row, col, depth, depths, parents, visited, low):
    visited[row, col] = 1
    depths[row,col] = depth
    low[row,col] = depth
    children = 0
    for n in grid_neighbors(row, col):
        nr,nc = n
        if state[nr,nc] != 0 and state[nr,nc] != ARTICULATION:
            continue
        if visited[nr,nc] == 0:
            parents[nr,nc] = id_mat(row,col)
            rec_hopcroft_tarjan(state, nr, nc, depth+1, depths, parents, visited, low)
            children += 1
            if (low[nr,nc] >= depths[row,col]) and (parents[row,col] != -1):
                state[row,col] = ARTICULATION
            low[row,col] = min(low[row,col], low[nr,nc])
        elif id_mat(nr,nc) != parents[row,col]:
            low[row,col] = min(low[row,col], depths[nr,nc])
    if (parents[row,col] == -1 and children >= 2):
        state[row,col] = ARTICULATION

def compute_voronoi(player, opponent, state):
    start_time = timeit.default_timer()
    head = player.pos
    ophead = opponent.pos
    player_costs = dijkstra(state, head)
    op_costs = dijkstra(state, ophead)
    
    maxcost = SIZE[0] + SIZE[1]
    mask_player = (player_costs < op_costs) & (player_costs <= maxcost)
    mask_op = (op_costs < player_costs) & (op_costs <= maxcost)
    v = float((np.sum(mask_player) - np.sum(mask_op)) / (SIZE[0]*SIZE[1]))

    # print(str(timeit.default_timer() - start_time))
    return v

def chamber_heuristic(walls, player, opponent):
    state = get_state(walls, player, opponent)
    hopcroft_tarjan(state)
    return compute_voronoi(player, opponent, state)


if __name__ == '__main__':
    from environment.tron import Tron
    tron = Tron(*SIZE)
    tron.reset()
    tron.tick((-1, 0), (1, 0))

    chamber = chamber_heuristic(tron.walls, tron.bike1, tron.bike2)
    print(chamber)