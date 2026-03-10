using UnityEngine;
using System.Collections.Generic;

public class BikeSprite : MonoBehaviour
{
    SpriteRenderer sprite;
    LineRenderer trail;
    void Awake()
    {
        sprite = GetComponentInChildren<SpriteRenderer>();
        trail = GetComponent<LineRenderer>();
        trail.startWidth = 0.5f;
        trail.endWidth = 0.5f;

    }

    public void Reset(int orientation)
    {
        Rotate(orientation);
        trail.positionCount = 0;
    }

    public void Rotate(int action)
    {
        // DIRS = { up, right, down, left }
        // Rotation angles for each direction (in degrees)
        float[] angles = { 0f, 270f, 180f, 90f };
        sprite.transform.rotation = Quaternion.Euler(0, 0, angles[action]);
    }

    public void AddTrail(Vector3 position)
    {
        trail.positionCount += 1;
        trail.SetPosition(trail.positionCount - 1, position + new Vector3(0.5f, 0.5f, 0));
    }

}
