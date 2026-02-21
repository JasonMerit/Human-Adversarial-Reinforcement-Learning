using UnityEngine;
using System.Collections.Generic;

public static class Heuristic
{
    const int FRIENDLY = 1;
    const int OPPONENT = 2;
    const int ARTICULATION = 3;

    static int width;
    static int height;

    public static float ChamberHeuristic(int[,] walls, Vector2Int player, Vector2Int opponent)
    {
        height = walls.GetLength(0);
        width  = walls.GetLength(1);

        int[,] state = GetState(walls, player, opponent);

        HopcroftTarjan(state);

        return ComputeVoronoi(player, opponent, state);
    }

    static int[,] GetState(int[,] walls, Vector2Int player, Vector2Int opponent)
    {
        int[,] state = (int[,])walls.Clone();

        state[player.y, player.x] = FRIENDLY;
        state[opponent.y, opponent.x] = OPPONENT;

        return state;
    }

    static IEnumerable<Vector2Int> GridNeighbors(int r, int c)
    {
        if (r + 1 < height) yield return new Vector2Int(c, r + 1);
        if (r - 1 >= 0)     yield return new Vector2Int(c, r - 1);
        if (c + 1 < width)  yield return new Vector2Int(c + 1, r);
        if (c - 1 >= 0)     yield return new Vector2Int(c - 1, r);
    }

    static float[,] Dijkstra(int[,] state, Vector2Int head)
    {
        float[,] dists = new float[height, width];

        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
                dists[r, c] = float.PositiveInfinity;

        Queue<(int r, int c, int dist)> q = new Queue<(int, int, int)>();

        q.Enqueue((head.y, head.x, 0));
        dists[head.y, head.x] = 0;

        while (q.Count > 0)
        {
            var (cr, cc, dist) = q.Dequeue();

            foreach (var n in GridNeighbors(cr, cc))
            {
                int nr = n.y;
                int nc = n.x;

                if (state[nr, nc] != 0) continue;
                if (dists[nr, nc] < float.PositiveInfinity) continue;

                dists[nr, nc] = dist + 1;
                q.Enqueue((nr, nc, dist + 1));
            }
        }

        return dists;
    }

    static void HopcroftTarjan(int[,] state)
    {
        int[,] parents = new int[height, width];
        bool[,] visited = new bool[height, width];
        int[,] depth = new int[height, width];
        int[,] low = new int[height, width];

        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
            {
                parents[r, c] = -1;
                depth[r, c] = -1;
                low[r, c] = -1;
            }

        RecHT(state, 0, 0, 0, parents, visited, depth, low);
    }

    static void RecHT(int[,] state, int r, int c, int d,
        int[,] parents, bool[,] visited, int[,] depth, int[,] low)
    {
        visited[r, c] = true;
        depth[r, c] = d;
        low[r, c] = d;
        int children = 0;

        foreach (var n in GridNeighbors(r, c))
        {
            int nr = n.y;
            int nc = n.x;

            if (state[nr, nc] != 0 && state[nr, nc] != ARTICULATION)
                continue;

            if (!visited[nr, nc])
            {
                parents[nr, nc] = r * width + c;
                children++;

                RecHT(state, nr, nc, d + 1, parents, visited, depth, low);

                if (low[nr, nc] >= depth[r, c] && parents[r, c] != -1)
                    state[r, c] = ARTICULATION;

                low[r, c] = Mathf.Min(low[r, c], low[nr, nc]);
            }
            else if (parents[r, c] != nr * width + nc)
            {
                low[r, c] = Mathf.Min(low[r, c], depth[nr, nc]);
            }
        }

        if (parents[r, c] == -1 && children >= 2)
            state[r, c] = ARTICULATION;
    }

    static float ComputeVoronoi(Vector2Int player, Vector2Int opponent, int[,] state)
    {
        float[,] playerCosts = Dijkstra(state, player);
        float[,] oppCosts = Dijkstra(state, opponent);

        int playerCount = 0;
        int oppCount = 0;

        int maxCost = width + height;

        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                float pc = playerCosts[r, c];
                float oc = oppCosts[r, c];

                if (pc < oc && pc <= maxCost)
                    playerCount++;
                else if (oc < pc && oc <= maxCost)
                    oppCount++;
            }
        }

        return (float)(playerCount - oppCount) / (width * height);
    }
}