using UnityEngine;

public class Adversary
{
    static readonly Vector2Int[] DIRS = 
    {
        new Vector2Int(0,1),   // Up
        new Vector2Int(1,0),   // Right
        new Vector2Int(0,-1),  // Down
        new Vector2Int(-1,0)   // Left
    };

    public static Vector2Int ChooseMove(
        int[,] walls,
        Vector2Int playerPos,
        Vector2Int adversaryPos)
    {
        int height = walls.GetLength(0);
        int width  = walls.GetLength(1);

        float bestScore = float.NegativeInfinity;
        Vector2Int bestMove = Vector2Int.zero;

        foreach (var dir in DIRS)
        {
            Vector2Int newPos = adversaryPos + dir;

            if (!IsLegal(newPos, walls, width, height))
                continue;

            int[,] simWalls = (int[,])walls.Clone();

            // Mark current adversary position as wall
            simWalls[adversaryPos.y, adversaryPos.x] = 2;

            float score = Heuristic.ChamberHeuristic(
                simWalls,
                playerPos,
                newPos
            );

            if (score > bestScore)
            {
                bestScore = score;
                bestMove = dir;
            }
        }

        return bestMove;
    }

    static bool IsLegal(Vector2Int pos, int[,] walls, int width, int height)
    {
        if (pos.x < 0 || pos.x >= width) return false;
        if (pos.y < 0 || pos.y >= height) return false;
        if (walls[pos.y, pos.x] != 0) return false;

        return true;
    }
}