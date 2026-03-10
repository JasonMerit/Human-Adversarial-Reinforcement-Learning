
using UnityEngine;

public static class Constants
{
    public static readonly Vector2Int[] DIRS = new Vector2Int[] {
        new Vector2Int(0, 1),   // Up
        new Vector2Int(1, 0),   // Right
        new Vector2Int(0, -1),  // Down
        new Vector2Int(-1, 0)   // Left
    };

    public static readonly int BOARD_SIZE = 25;
    public static readonly Color cyan = new(0f, 0.9f, 0.8f);
    public static readonly Color orange = new(1f, 0.6f, 0f);
    public static readonly Color green = new(0.04f, 0.2f, 0.04f);
    public static readonly Color green_alt = new(0.04f, 0.25f, 0.04f);
}