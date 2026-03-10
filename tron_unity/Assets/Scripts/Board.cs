using UnityEngine;
using UnityEngine.Tilemaps;

public class Board : MonoBehaviour
{
    [SerializeField] TileBase tile;
    Tilemap tilemap;

    void Awake()
    {
        tilemap = GetComponentInChildren<Tilemap>();
        Reset();
    }

    public void SetCell(Vector2Int cell, Color color)
    {
        Vector3Int pos = (Vector3Int)cell;
        tilemap.SetTile(pos, tile);
        tilemap.SetColor(pos, color);
    }

    public void Reset()
    {
        tilemap.ClearAllTiles();
        // Draw checkered pattern for visibility
        for (int x = 0; x < Constants.BOARD_SIZE; x++)
        {
            for (int y = 0; y < Constants.BOARD_SIZE; y++)
            {
                Color color = (x + y) % 2 == 0 ? Constants.green : Constants.green_alt;
                SetCell(new Vector2Int(x, y), color);
            }
        }
    }
}