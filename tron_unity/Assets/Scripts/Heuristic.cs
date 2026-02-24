using System.Collections.Generic;
using UnityEngine;

public static class Heuristic
{
    public static readonly Vector2Int[] DIRS = { Vector2Int.up, Vector2Int.right, Vector2Int.down, Vector2Int.left };

    public static float ChamberHeuristic(int[,] walls, Vector2Int you, Vector2Int other)
    {
        return 0f; // placeholder for testing
    }

    public static int[,] FillBoard(int[,] board, Vector2Int p1, Vector2Int p2)
    {
        int width = board.GetLength(0);
        int height = board.GetLength(1);

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
        FloodFill(board, p1, dist1);

        // BFS for Player 2
        FloodFill(board, p2, dist2);

        int region1 = 0;
        int region2 = 0;

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                if (board[x, y] != 0)
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
    
    public static int ComputeVoronoiScore(int[,] board, Vector2Int p1, Vector2Int p2)
    {
        int width = board.GetLength(0);
        int height = board.GetLength(1);

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
        FloodFill(board, p1, dist1);

        // BFS for Player 2
        FloodFill(board, p2, dist2);

        int region1 = 0;
        int region2 = 0;

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                if (board[x, y] != 0)
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

    static void FloodFill(int[,] board, Vector2Int start, int[,] dist)
    {
        int width = board.GetLength(0);
        int height = board.GetLength(1);

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

                if (board[next.x, next.y] != 0)
                    continue;

                if (dist[next.x, next.y] != int.MaxValue)
                    continue;

                dist[next.x, next.y] = dist[current.x, current.y] + 1;
                queue.Enqueue(next);
            }
        }
    }
}