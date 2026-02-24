using UnityEngine;
using UnityEngine.Tilemaps;

public class Board : MonoBehaviour
{
    [SerializeField] TileBase tile;
    Tilemap tilemap;

    void Awake()
    {
        tilemap = GetComponentInChildren<Tilemap>();
    }

    public void SetCell(Vector2Int cell, Color color)
    {
        Vector3Int pos = (Vector3Int)cell;
        tilemap.SetTile(pos, tile);
        tilemap.SetColor(pos, color);
    }

    public void Clear()
    {
        tilemap.ClearAllTiles();
    }
}