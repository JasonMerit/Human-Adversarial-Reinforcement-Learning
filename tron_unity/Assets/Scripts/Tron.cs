using Unity.VisualScripting;
using UnityEngine;

public class TronBike
{
    public Vector2Int pos;
    public TronBike(Vector2Int startPos)
    {
        pos = startPos;
    }

    // Returns true if crash
    public bool Move(Vector2Int vel, int[,] trails)
    {
        pos += vel;
        pos.x = Mathf.Clamp(pos.x, 0, trails.GetLength(0) - 1);
        pos.y = Mathf.Clamp(pos.y, 0, trails.GetLength(1) - 1);
        return IsHit(trails, pos.x, pos.y);
    }

    public bool IsHitInDir(int[,] trails, Vector2Int dir)
    {
        Vector2Int newPos = pos + dir;
        return IsHit(trails, newPos.x, newPos.y);
    }

    public bool IsHit(int[,] trails, int x, int y)
    {
        return trails[x, y] != 0;
    }
}

public enum GameState { Draw, Bike1Win, Bike2Win, Playing }
public class Tron
{
    public int width;
    public int height;

    public int[,] trails;

    public TronBike bike1;
    public TronBike bike2;

    public Tron(Vector2Int size)
    {
        width = size.x;
        height = size.y;
    }

    public void Reset()
    {
        trails = new int[width, height];
        bike1 = new TronBike(new Vector2Int(width / 6, height / 2));
        bike2 = new TronBike(new Vector2Int(5 * width / 6, height / 2));
    }

    // Returns:
    // -1 - No collision
    // 2 - Bike1 collided
    // 1 - Bike2 collided
    // 0 - Both bikes collided (draw)
    public GameState Step(Vector2Int dir1, Vector2Int dir2)
    {
        // Mark current positions as trails
        trails[bike1.pos.x, bike1.pos.y] = 1;
        trails[bike2.pos.x, bike2.pos.y] = 2;

        // Move bikes
        bool bike1Hit = bike1.Move(dir1, trails);
        bool bike2Hit = bike2.Move(dir2, trails);

        // Head-to-head collision
        if ((bike1Hit && bike2Hit) || bike1.pos == bike2.pos)
            return GameState.Draw;

        if (bike1Hit)
            return GameState.Bike2Win;

        if (bike2Hit)
            return GameState.Bike1Win;


        return GameState.Playing;
    }
}