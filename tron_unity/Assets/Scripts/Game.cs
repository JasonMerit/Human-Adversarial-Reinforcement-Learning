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
 
    // Worker worker;
    NetworkManager networkManager;
    PlayerInput playerInput;
    Tron tron;

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
        networkManager = GetComponent<NetworkManager>();

        playerColor = Constants.cyan;
        adversaryColor = Constants.orange;
        tron = new Tron(new Vector2Int(25, 25));

        // adversary.InitializeWorker(ModelLoader.Load(modelAsset));
    }

    void Start()
    {
        networkManager.DownloadONNXModel((path) => {
            if (string.IsNullOrEmpty(path)) Debug.LogError("Failed to download ONNX model.");
            else
            {
                Debug.Log($"ONNX model downloaded to: {path}");
                // Initialize the worker with the downloaded model
                Model model = ModelLoader.Load(path);
                adversary.InitializeWorker(model);
            }
        });
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
        
        // 2. Turn window into input buffer (update currentAction)
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

        // 3. Step simulation (Update bike positions)
        time += Time.deltaTime;
        if (time >= tickRate)
        {
            time -= tickRate;
            Step(currentAction); // advance tile
            commited = false;            
        }

        // 4. Interpolate for smooth visuals and border based end points
        float alpha = time / tickRate;
        Vector2 to = tron.bike1.pos + DIRS[currentAction] * 0.5f;

        player.LerpPosition(from, to, alpha);
        adversary.LerpPosition(advFrom, advTo, alpha);
    }
    
    void Step(int action)
    {
        int advAction = adversary.GetAction(tron.trails, tron.bike2.pos, tron.bike1.pos);
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
}
