using UnityEngine;
using Unity.InferenceEngine;
using System.Collections.Generic;
using UnityEngine.InputSystem;
using TMPro;

public class Game : MonoBehaviour
{
    public static readonly Vector2[] DIRS = { Vector2Int.up, Vector2Int.right, Vector2Int.down, Vector2Int.left };
    public readonly Vector2Int[] IDIRS = { Vector2Int.up, Vector2Int.right, Vector2Int.down, Vector2Int.left };

    [SerializeField] Board board;
    [SerializeField] Bike player;
    [SerializeField] Bike adversary;
    [SerializeField] ModelAsset modelAsset;
    [SerializeField] TMP_Text outputText;
    [SerializeField] CameraShake cameraShake;
 
    Worker worker;
    NetworkManager networkManager;
    Tron tron;
    PlayerInput playerInput;

    [HideInInspector] public GameState State;
    public float tickRate = 0.15f; //.15 seconds per tick
    public float renderOffset = .6f;
    List<Vector2Int> history = new();

    float time;
    [HideInInspector] public Color playerColor;
    [HideInInspector] public Color adversaryColor;

    // Input / action / lerping
    int currentAction;
    Queue<int> inputQueue = new Queue<int>(2);
    bool commited;
    Vector2 from;
    Vector2 advFrom;
    Vector2 advTo;

    void Awake()
    {
        playerInput = GetComponent<PlayerInput>();
        worker = new Worker(ModelLoader.Load(modelAsset), BackendType.GPUPixel);

        networkManager = GetComponent<NetworkManager>();
        playerColor = Constants.cyan;
        adversaryColor = Constants.orange;
        tron = new Tron(new Vector2Int(25, 25));

        // Call RunInference(); repeatedly to test
        // InvokeRepeating(nameof(RunInference), 0f, 1f);
    }

    public void Reset()
    {
        State = GameState.Playing;
        time = 0;

        tron.Reset();
        board.Reset();
        
        currentAction = 1;
        inputQueue.Clear();
        commited = false;

        from = tron.bike1.pos - DIRS[currentAction] * renderOffset;
        advFrom = tron.bike2.pos - DIRS[3] * renderOffset;
        advTo = advFrom + DIRS[3];

        player.Reset(1, from);
        adversary.Reset(3, advFrom);
    }

    public void Tick()
    {       
        //  1. Get input 
        if (playerInput.actions["Up"].triggered) inputQueue.Enqueue(0);
        else if (playerInput.actions["Right"].triggered) inputQueue.Enqueue(1);
        else if (playerInput.actions["Down"].triggered) inputQueue.Enqueue(2);
        else if (playerInput.actions["Left"].triggered) inputQueue.Enqueue(3);
        
        // 3. Turn window into input buffer (update currentAction)
        if (!commited && inputQueue.Count > 0)
        {
            int next = inputQueue.Dequeue();
            if (next != currentAction && (next + 2) % 4 != currentAction) // prevent reverse
            {
                currentAction = next;
                commited = true;
                player.Rotate(currentAction);
            }
        }

        // 4. Step simulation (Update bike positions)
        time += Time.deltaTime;
        if (time >= tickRate)
        {
            time -= tickRate;
            Step(currentAction); // advance tile
            commited = false;            
        }

        // 5. Interpolate for smooth visuals and border based end points
        float alpha = time / tickRate;
        Vector2 to = tron.bike1.pos + DIRS[currentAction] * 0.5f;

        player.LerpPosition(from, to, alpha);
        adversary.LerpPosition(advFrom, advTo, alpha);
    }
    
    void Step(int action)
    {
        int advAction = GetAction(worker, adversary.orientation, tron.trails, tron.bike2.pos, tron.bike1.pos);
        int advHeading = (adversary.orientation + (advAction - 1) + 4) % 4;
        // int advAction = Adversary.ChooseMove(tron.trails, tron.bike2.pos, tron.bike1.pos);
        int playerAction = action;
        history.Add(new (playerAction, advHeading));

        State = tron.Step(IDIRS[playerAction], IDIRS[advHeading]);
        if (State != GameState.Playing) { EndEpisode(playerAction); return; }

        // Rendering and lerping setup
        player.Transform(playerAction, tron.bike1.pos);
        adversary.Transform(advHeading, tron.bike2.pos);

        from = tron.bike1.pos - DIRS[currentAction] * renderOffset;
        advFrom = tron.bike2.pos - DIRS[advHeading] * renderOffset;
        advTo = advFrom + DIRS[advHeading];
    }

    void EndEpisode(int playerAction)
    {
        if (State == GameState.Bike2Win) {
            cameraShake.Shake(IDIRS[playerAction]);
            player.Crash();
        }
        else if (State == GameState.Bike1Win) {
            cameraShake.Shake();
            adversary.Crash();
        }
        else
        {
            cameraShake.Shake();
            player.Crash();
            adversary.Crash();
        }

        bool trapped = IsTrapped(tron.bike1);

        if (Main.PostingEnabled) { networkManager.SendEpisode(history, (int)State, trapped); }
        history = new();
    }

    bool IsTrapped(TronBike bike)
    {
        foreach (var dir in IDIRS) { if (!bike.IsHitInDir(tron.trails, dir)) return false; }
        return true;
    }

    (int tx, int ty) Rotate(int x, int y, int size, int o)
    {
        if (o == 0) return (x, y);  // Up
        if (o == 1) return (size - 1 - y, x);  // Right
        if (o == 2) return (size - 1 - x, size - 1 - y);  // Down
        return (y, size - 1 - x);  // Left
    }

    int GetAction(Worker worker, int orientation, int[,] trails, Vector2 you, Vector2 other)
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
