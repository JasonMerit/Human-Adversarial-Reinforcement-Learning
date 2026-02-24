using System;
using System.Collections.Generic;
using UnityEngine;

public static class Heuristic
{
    public static readonly Vector2Int[] DIRS = { Vector2Int.up, Vector2Int.right, Vector2Int.down, Vector2Int.left };

    public static int[,] FillBoard(int[,] trails, Vector2Int p1, Vector2Int p2)
    {
        int width = trails.GetLength(0);
        int height = trails.GetLength(1);

        int[,] ownership = new int[width, height];
        int[,] dist1 = new int[width, height];
        int[,] dist2 = new int[width, height];

        const int INF = int.MaxValue;

        // Initialize distances
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                dist1[x, y] = INF;
                dist2[x, y] = INF;
            }
        }

        // BFS for Player 1
        FloodFill(trails, p1, dist1);

        // BFS for Player 2
        FloodFill(trails, p2, dist2);

        int region1 = 0;
        int region2 = 0;

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                if (trails[x, y] != 0)
                    continue;

                int d1 = dist1[x, y];
                int d2 = dist2[x, y];

                if (d1 == INF && d2 == INF) {
                    // Debug.Log($"Unreachable free cell at {x},{y}");
                    continue;
                }

                if (d1 < d2) {
                    region1++;
                    ownership[x, y] = 1;
                }
                else if (d2 < d1) {
                    region2++;
                    ownership[x, y] = 2;
                }
                else {
                    ownership[x, y] = 3; // battlefront
                }
            }
        }

        return ownership;
    }
    
    public static int Voronoi(int[,] trails, Vector2Int p1, Vector2Int p2)
    {
        int width = trails.GetLength(0);
        int height = trails.GetLength(1);

        int[,] dist1 = new int[width, height];
        int[,] dist2 = new int[width, height];

        const int INF = int.MaxValue;

        // Initialize distances
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                dist1[x, y] = INF;
                dist2[x, y] = INF;
            }
        }

        // BFS for Player 1
        FloodFill(trails, p1, dist1);

        // BFS for Player 2
        FloodFill(trails, p2, dist2);

        int region1 = 0;
        int region2 = 0;

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                if (trails[x, y] != 0)
                    continue;

                int d1 = dist1[x, y];
                int d2 = dist2[x, y];

                if (d1 == INF && d2 == INF)
                    continue;

                if (d1 < d2)
                    region1++;
                else if (d2 < d1)
                    region2++;
                // equal distance = battlefront → ignored
            }
        }

        return region1 - region2;
    }

    static void FloodFill(int[,] trails, Vector2Int start, int[,] dist)
    {
        int width = trails.GetLength(0);
        int height = trails.GetLength(1);

        Queue<Vector2Int> queue = new Queue<Vector2Int>();

        dist[start.x, start.y] = 0;
        queue.Enqueue(start);

        while (queue.Count > 0)
        {
            Vector2Int current = queue.Dequeue();
            foreach (var dir in DIRS)
            {
                Vector2Int next = current + dir;

                if (next.x < 0 || next.x >= width ||
                    next.y < 0 || next.y >= height)
                    continue;

                if (trails[next.x, next.y] != 0)
                    continue;

                if (dist[next.x, next.y] != int.MaxValue)
                    continue;

                dist[next.x, next.y] = dist[current.x, current.y] + 1;
                queue.Enqueue(next);
            }
        }
    }

    public static float Chamber(int[,] trails, Vector2Int you, Vector2Int other)
    {
        int width = trails.GetLength(0);
        int height = trails.GetLength(1);

        // Compute Voronoi distances
        int[,] distYou = new int[width, height];
        int[,] distOther = new int[width, height];
        const int INF = int.MaxValue;

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                distYou[x, y] = distOther[x, y] = INF;

        FloodFill(trails, you, distYou);
        FloodFill(trails, other, distOther);

        // Determine Voronoi region of "you"
        bool[,] voronoiYou = new bool[width, height];
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                if (trails[x, y] != 0) continue;

                int d1 = distYou[x, y];
                int d2 = distOther[x, y];

                if (d1 == INF) continue;

                if (d1 < d2) voronoiYou[x, y] = true;
            }
        }

        // If no articulation points exist in Voronoi region, return its size
        var articulation = FindArticulationPoints(voronoiYou, width, height);
        if (articulation.Count == 0)
            return CountCells(voronoiYou, width, height);

        // Build chambers
        bool[,] visited = new bool[width, height];
        List<int> chamberSizes = new List<int>();

        foreach (var ap in articulation)
        {
            foreach (var dir in DIRS)
            {
                Vector2Int start = ap + dir;

                if (!InBounds(start, width, height)) continue;
                if (!voronoiYou[start.x, start.y]) continue;
                if (articulation.Contains(start)) continue;
                if (visited[start.x, start.y]) continue;

                int size = FloodChamber(voronoiYou, articulation, visited, start, width, height);
                if (size > 0)
                    chamberSizes.Add(size);
            }
        }

        int youValue = CountCells(voronoiYou, width, height);

        if (chamberSizes.Count > 0)
        {
            int maxChamber = 0;
            foreach (int s in chamberSizes)
                if (s > maxChamber) maxChamber = s;

            youValue = CountChamber(visited, width, height) + maxChamber;
        }

        // Compute opponent value symmetrically
        return youValue - ComputeOpponentChamberValue(trails, you, other, distYou, distOther);
    }

    /* ---------------- Helpers ---------------- */

    static int FloodChamber(bool[,] region, HashSet<Vector2Int> articulation,
                            bool[,] visited, Vector2Int start,
                            int width, int height)
    {
        int count = 0;
        Queue<Vector2Int> q = new Queue<Vector2Int>();
        q.Enqueue(start);
        visited[start.x, start.y] = true;

        while (q.Count > 0)
        {
            var cur = q.Dequeue();
            count++;

            foreach (var dir in DIRS)
            {
                var next = cur + dir;
                if (!InBounds(next, width, height)) continue;
                if (!region[next.x, next.y]) continue;
                if (articulation.Contains(next)) continue;
                if (visited[next.x, next.y]) continue;

                visited[next.x, next.y] = true;
                q.Enqueue(next);
            }
        }

        return count;
    }

    static int CountCells(bool[,] region, int width, int height)
    {
        int count = 0;
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                if (region[x, y]) count++;
        return count;
    }

    static bool InBounds(Vector2Int p, int width, int height)
    {
        return p.x >= 0 && p.x < width && p.y >= 0 && p.y < height;
    }

    static HashSet<Vector2Int> FindArticulationPoints(bool[,] region, int width, int height)
    {
        // Standard DFS Tarjan for grid graph
        HashSet<Vector2Int> result = new HashSet<Vector2Int>();
        int[,] disc = new int[width, height];
        int[,] low = new int[width, height];
        bool[,] visited = new bool[width, height];
        int time = 0;

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                disc[x, y] = -1;

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                if (region[x, y] && disc[x, y] == -1)
                    APDFS(new Vector2Int(x, y), null);
            }
        }

        return result;

        void APDFS(Vector2Int u, Vector2Int? parent)
        {
            disc[u.x, u.y] = low[u.x, u.y] = time++;
            int children = 0;

            foreach (var dir in DIRS)
            {
                var v = u + dir;
                if (!InBounds(v, width, height)) continue;
                if (!region[v.x, v.y]) continue;

                if (disc[v.x, v.y] == -1)
                {
                    children++;
                    APDFS(v, u);
                    low[u.x, u.y] = Math.Min(low[u.x, u.y], low[v.x, v.y]);

                    if (parent != null && low[v.x, v.y] >= disc[u.x, u.y])
                        result.Add(u);
                }
                else if (parent == null || v != parent)
                {
                    low[u.x, u.y] = Math.Min(low[u.x, u.y], disc[v.x, v.y]);
                }
            }

            if (parent == null && children > 1)
                result.Add(u);
        }
    }

    static int CountChamber(bool[,] visited, int width, int height)
    {
        int count = 0;
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                if (visited[x, y])
                    count++;
        return count;
    }

    static float ComputeOpponentChamberValue(
        int[,] walls,
        Vector2Int you,
        Vector2Int other,
        int[,] distYou,
        int[,] distOther)
    {
        int width = walls.GetLength(0);
        int height = walls.GetLength(1);

        const int INF = int.MaxValue;

        // Build opponent Voronoi region
        bool[,] voronoiOther = new bool[width, height];

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                if (walls[x, y] != 0) continue;

                int d1 = distYou[x, y];
                int d2 = distOther[x, y];

                if (d2 == INF) continue;

                if (d2 < d1)
                    voronoiOther[x, y] = true;
            }
        }

        var articulation = FindArticulationPoints(voronoiOther, width, height);

        if (articulation.Count == 0)
            return CountCells(voronoiOther, width, height);

        bool[,] visited = new bool[width, height];
        List<int> chamberSizes = new List<int>();

        foreach (var ap in articulation)
        {
            foreach (var dir in DIRS)
            {
                Vector2Int start = ap + dir;

                if (!InBounds(start, width, height)) continue;
                if (!voronoiOther[start.x, start.y]) continue;
                if (articulation.Contains(start)) continue;
                if (visited[start.x, start.y]) continue;

                int size = FloodChamber(
                    voronoiOther,
                    articulation,
                    visited,
                    start,
                    width,
                    height);

                if (size > 0)
                    chamberSizes.Add(size);
            }
        }

        int opponentValue = CountCells(voronoiOther, width, height);

        if (chamberSizes.Count > 0)
        {
            int maxChamber = 0;
            foreach (int s in chamberSizes)
                if (s > maxChamber)
                    maxChamber = s;

            opponentValue = CountChamber(visited, width, height) + maxChamber;
        }

        return opponentValue;
    }
}