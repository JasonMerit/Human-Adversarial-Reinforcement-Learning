using UnityEngine;

// Assumes player stands still and adversary moves one step.
// Evaluates the resulting state using a Voronoi heuristic.

public class Adversary
{
    static readonly Vector2Int[] DIRS = 
    {
        new(0,1),   // Up
        new(1,0),   // Right
        new(0,-1),  // Down
        new(-1,0)   // Left
    };

    public static int ChooseMove(int[,] walls, Vector2Int you, Vector2Int other) {
        float bestScore = float.NegativeInfinity;
        int bestAction = 0;
        int[,] simWalls = (int[,])walls.Clone();
        simWalls[you.y, you.x] = 1;  // Mark current position as wall

        for (int i = 0; i < DIRS.Length; i++) {
            Vector2Int newPos = you + DIRS[i];;
            if (!IsLegal(newPos, walls)) continue;

            float score = Heuristic.ChamberHeuristic(simWalls, newPos, other);

            if (score > bestScore) {
                bestScore = score;
                bestAction = i;
            }
        }
        return bestAction;
    }

    static bool IsLegal(Vector2Int pos, int[,] walls) {
        if (pos.x < 0 || pos.x >= walls.GetLength(1)) return false;
        if (pos.y < 0 || pos.y >= walls.GetLength(0)) return false;
        if (walls[pos.y, pos.x] != 0) return false;
        return true;
    }
}