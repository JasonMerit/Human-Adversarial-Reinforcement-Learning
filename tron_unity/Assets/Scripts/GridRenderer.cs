using UnityEngine;

public class GridRenderer : MonoBehaviour
{
    public float lineWidth = .1f;
    public Material lineMaterial;

    void Start()
    {
        DrawGrid();
    }

    void DrawGrid()
    {
        float gridSize = Constants.BOARD_SIZE;

        // Vertical lines
        for (int x = 0; x <= Constants.BOARD_SIZE; x++)
        {
            Vector3 start = new Vector3(x, 0, 0);
            Vector3 end   = new Vector3(x, gridSize, 0);

            CreateLine(start, end);
        }

        // Horizontal lines
        for (int y = 0; y <= Constants.BOARD_SIZE; y++)
        {
            Vector3 start = new Vector3(0, y, 0);
            Vector3 end   = new Vector3(gridSize, y, 0);

            CreateLine(start, end);
        }
    }

    void CreateLine(Vector3 start, Vector3 end)
    {
        GameObject line = new GameObject("GridLine");
        line.transform.parent = transform;

        LineRenderer lr = line.AddComponent<LineRenderer>();
        lr.material = lineMaterial;
        lr.startWidth = lineWidth;
        lr.endWidth = lineWidth;

        lr.positionCount = 2;
        lr.SetPosition(0, start);
        lr.SetPosition(1, end);
        
        lr.sortingLayerName = "Background";
        lr.sortingOrder = 1;  // Infront of tilemap

        lr.useWorldSpace = false;
    }
}