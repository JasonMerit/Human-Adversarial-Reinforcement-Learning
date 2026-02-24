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

    public static int ChooseMove(int[,] trails, Vector2Int you, Vector2Int other) {
        float bestScore = float.NegativeInfinity;
        int bestAction = 0;
        // simWalls[you.y, you.x] = 1;  // Mark current position as wall

        for (int i = 0; i < DIRS.Length; i++) {
            Vector2Int newPos = you + DIRS[i];;
            if (!IsLegal(newPos, trails)) continue;

            // float score = Heuristic.Chamber(trails, newPos, other);
            float score = Heuristic.Voronoi(trails, newPos, other);

            if (score > bestScore) {
                bestScore = score;
                bestAction = i;
            }
        }
        return bestAction;
    }

    static bool IsLegal(Vector2Int pos, int[,] trails) {
        if (pos.x < 0 || pos.x >= trails.GetLength(1)) return false;
        if (pos.y < 0 || pos.y >= trails.GetLength(0)) return false;
        if (trails[pos.x, pos.y] != 0) return false;
        return true;
    }
}