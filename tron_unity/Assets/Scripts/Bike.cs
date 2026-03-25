using UnityEngine;

public class Bike : MonoBehaviour
{
    SpriteRenderer sprite;
    LineRenderer trail;
    ParticleSystem crashParticles;

    public int orientation; // 0=up, 1=right, 2=down, 3=left
    
    void Awake()
    {
        sprite = GetComponentInChildren<SpriteRenderer>();
        trail = GetComponent<LineRenderer>();
        crashParticles = GetComponentInChildren<ParticleSystem>();
        trail.startWidth = .5f;
        trail.endWidth = .5f;
    }

    public void Reset(int orientation, Vector2 position)
    {
        transform.position = (Vector3)position;
        trail.positionCount = 0;
        sprite.enabled = true;
        crashParticles.Clear();
        crashParticles.Stop();

        // Provided non-center position, so center it and add initial trail point
        position += Game.DIRS[orientation];
        position = new Vector2(Mathf.Round(position.x), Mathf.Round(position.y));
        Transform(orientation, position);
    }

    public void Transform(int action, Vector2 position)
    {
        Rotate(action);
        AddTrail(position);
    }

    public void AddTrail(Vector2 position)
    {
        Vector3 newPoint = (Vector3)position + new Vector3(0.5f, 0.5f, 0);
        trail.positionCount += 1;
        trail.SetPosition(trail.positionCount - 1, newPoint);
    }

    public void Rotate(int action)
    {
        // DIRS = { up, right, down, left }
        // Rotation angles for each direction (in degrees)
        float[] angles = { 0f, 270f, 180f, 90f };
        orientation = action;
        sprite.transform.rotation = Quaternion.Euler(0, 0, angles[action]);
    }

    public void Crash()
    {
        sprite.enabled = false;
        crashParticles.Play();
    }

    public void LerpPosition(Vector2 from, Vector2 to, float t)
    {
        transform.position = Vector2.Lerp(from, to, t);
    }

}
