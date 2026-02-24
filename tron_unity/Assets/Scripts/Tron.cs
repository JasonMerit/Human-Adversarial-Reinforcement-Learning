using Unity.VisualScripting;
using UnityEngine;

public class Bike
{
    public Vector2Int pos;
    public Vector2Int lastPos;
    public Bike(Vector2Int startPos)
    {
        pos = startPos;
        lastPos = startPos;
    }

    // Returns true if crash
    public bool Move(Vector2Int vel, int[,] trails)
    {
        Vector2Int newPos = pos + vel;

        if (IsHit(trails, newPos.x, newPos.y))
            return true;
        
        lastPos = pos;
        pos = newPos;
        return false;
    }

    public bool IsHitInDir(int[,] trails, Vector2Int dir)
    {
        Vector2Int newPos = pos + dir;
        return IsHit(trails, newPos.x, newPos.y);
    }

    public bool IsHit(int[,] trails, int x, int y)
    {
        int height = trails.GetLength(0);
        int width  = trails.GetLength(1);

        if (y < 0 || y >= height) return true;
        if (x < 0 || x >= width)  return true;
        if (trails[x, y] != 0)     return true;

        return false;
    }
}

public enum GameState { Draw, Bike1Win, Bike2Win, Playing }
public class Tron
{
    public int width;
    public int height;

    public int[,] trails;

    public Bike bike1;
    public Bike bike2;

    public Tron(Vector2Int size)
    {
        width = size.x;
        height = size.y;
    }

    public void Reset()
    {
        trails = new int[width, height];

        bike1 = new Bike(new Vector2Int(1, height / 2));
        bike2 = new Bike(new Vector2Int(width - 2, height / 2));
        trails[bike1.pos.x, bike1.pos.y] = 1;
        trails[bike2.pos.x, bike2.pos.y] = 2;

    }

    // Returns:
    // -1 - No collision
    // 2 - Bike1 collided
    // 1 - Bike2 collided
    // 0 - Both bikes collided (draw)
    public GameState Step(Vector2Int dir1, Vector2Int dir2)
    {
        // Mark current positions as trails

        bool bike1Hit = bike1.Move(dir1, trails);
        bool bike2Hit = bike2.Move(dir2, trails);

        // Head-to-head collision
        if ((bike1Hit && bike2Hit) || bike1.pos == bike2.pos)
            return GameState.Draw;

        if (bike1Hit)
            return GameState.Bike2Win;

        if (bike2Hit)
            return GameState.Bike1Win;

        trails[bike1.pos.x, bike1.pos.y] = 1;
        trails[bike2.pos.x, bike2.pos.y] = 2;

        return GameState.Playing;
    }
}