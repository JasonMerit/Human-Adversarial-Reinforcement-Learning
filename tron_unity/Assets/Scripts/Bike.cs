using UnityEngine;
using Unity.InferenceEngine;

public class Bike : MonoBehaviour
{
    SpriteRenderer sprite;
    LineRenderer trail;
    ParticleSystem crashParticles;

    public int orientation; // 0=up, 1=right, 2=down, 3=left
    Worker worker;
    
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

    // =========================
    // ======== Sentis =========
    // =========================

    public void InitializeWorker(Model model)
    {
        worker = new Worker(model, BackendType.GPUPixel);
    }

    (int tx, int ty) Rotate(int x, int y, int size, int o)
    {
        if (o == 0) return (x, y);  // Up
        if (o == 1) return (size - 1 - y, x);  // Right
        if (o == 2) return (size - 1 - x, size - 1 - y);  // Down
        return (y, size - 1 - x);  // Left
    }

    public int GetAction(int[,] trails, Vector2 you, Vector2 other)
    {
        Tensor<float> input = new Tensor<float>(new TensorShape(1, 25, 25, 3));

        // Fill NHWC tensor explicitly (Sentis expects NHWC by default)
        for (int x = 0; x < 25; x++) {
            for (int y = 0; y < 25; y++) {
                var (tx, ty) = Rotate(x, y, 25, orientation);
                int fy = 24 - ty;  // Numpy origin is top-left, Unity's is bottom-left

                input[0, fy, tx, 0] = (trails[x, y] != 0) ? 1f : 0f;
                input[0, fy, tx, 1] = (you.x == x && you.y == y) ? 1f : 0f;
                input[0, fy, tx, 2] = (other.x == x && other.y == y) ? 1f : 0f;
            }
        }

        worker.Schedule(input);
        using var output = (worker.PeekOutput() as Tensor<float>).ReadbackAndClone();

        // Expected shape: (1, 1, 1, n_actions)
        int nActions = 3;

        float maxQ = float.NegativeInfinity;
        int bestAction = 0;

        for (int i = 0; i < nActions; i++)
        {
            float q = output[0, 0, 0, i];
            if (q > maxQ)
            {
                maxQ = q;
                bestAction = i;
            }
        }


        input.Dispose();
        return bestAction;
    }

    private void OnApplicationQuit()
    {
        worker?.Dispose();
    }

}
