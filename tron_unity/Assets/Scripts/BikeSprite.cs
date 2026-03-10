using UnityEngine;

public class BikeSprite : MonoBehaviour
{
    SpriteRenderer sprite;

    void Awake()
    {
        sprite = GetComponentInChildren<SpriteRenderer>();
    }

    public void Rotate(int action)
    {
        // DIRS = { up, right, down, left }
        // Rotation angles for each direction (in degrees)
        float[] angles = { 0f, 270f, 180f, 90f };
        sprite.transform.rotation = Quaternion.Euler(0, 0, angles[action]);
    }

}
