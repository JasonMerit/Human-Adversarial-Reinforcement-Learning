using UnityEngine;
using System.Collections.Generic;
using System;

public class Bike : MonoBehaviour
{
    SpriteRenderer sprite;
    LineRenderer trail;
    ParticleSystem crashParticles;
    
    int lastAction;
    
    void Awake()
    {
        sprite = GetComponentInChildren<SpriteRenderer>();
        trail = GetComponent<LineRenderer>();
        crashParticles = GetComponentInChildren<ParticleSystem>();
        float kek = .5f;
        trail.startWidth = kek;
        trail.endWidth = kek;
    }

    public void Reset(int orientation)
    {
        Rotate(orientation);
        trail.positionCount = 0;
        sprite.enabled = true;
        crashParticles.Clear();
        crashParticles.Stop();

        // Player specific
        lastAction = orientation;
    }

    public void Transform(int action, Vector2 position)
    {
        Rotate(action);

        Vector3 newPoint = (Vector3)position + new Vector3(0.5f, 0.5f, 0);
        trail.positionCount += 1;
        trail.SetPosition(trail.positionCount - 1, newPoint);
        lastAction = action;
    }

    public void Rotate(int action)
    {
        // DIRS = { up, right, down, left }
        // Rotation angles for each direction (in degrees)
        float[] angles = { 0f, 270f, 180f, 90f };
        sprite.transform.rotation = Quaternion.Euler(0, 0, angles[action]);
    }

    public void Crash()
    {
        sprite.enabled = false;
        crashParticles.Play();
    }

}
