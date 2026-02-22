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
    public bool Move(Vector2Int vel, int[,] walls)
    {
        Vector2Int newPos = pos + vel;

        if (IsHit(walls, newPos.x, newPos.y))
            return true;
        
        lastPos = pos;
        pos = newPos;
        return false;
    }

    public bool IsHit(int[,] walls, int x, int y)
    {
        int height = walls.GetLength(0);
        int width  = walls.GetLength(1);

        if (y < 0 || y >= height) return true;
        if (x < 0 || x >= width)  return true;
        if (walls[y, x] != 0)     return true;

        return false;
    }
}

public class Tron
{
    public int width;
    public int height;

    public int[,] walls;

    public Bike bike1;
    public Bike bike2;

    public Tron(Vector2Int size)
    {
        width = size.x;
        height = size.y;
    }

    public void Reset()
    {
        walls = new int[height, width];

        bike1 = new Bike(new Vector2Int(1, height / 2));
        bike2 = new Bike(new Vector2Int(width - 2, height / 2));
    }

    // Returns:
    // -1 - No collision
    // 2 - Bike1 collided
    // 1 - Bike2 collided
    // 0 - Both bikes collided (draw)
    public int Step(Vector2Int dir1, Vector2Int dir2)
    {
        // Mark current positions as walls
        walls[bike1.pos.y, bike1.pos.x] = 1;
        walls[bike2.pos.y, bike2.pos.x] = 2;

        bool bike1Hit = bike1.Move(dir1, walls);
        bool bike2Hit = bike2.Move(dir2, walls);

        // Head-to-head collision
        if ((bike1Hit && bike2Hit) || bike1.pos == bike2.pos)
            return 0;

        if (bike1Hit)
            return 2;

        if (bike2Hit)
            return 1;

        return -1;
    }
}