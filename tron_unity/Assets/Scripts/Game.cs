using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;
using UnityEngine.InputSystem;
using TMPro;

public class Game : MonoBehaviour
{
    public readonly Vector2[] DIRS = { Vector2Int.up, Vector2Int.right, Vector2Int.down, Vector2Int.left };
    public readonly Vector2Int[] IDIRS = { Vector2Int.up, Vector2Int.right, Vector2Int.down, Vector2Int.left };

    [SerializeField] Board board;
    [SerializeField] Bike player;
    [SerializeField] Bike adversary;
    [SerializeField] NNModel modelAsset;
    [SerializeField] TMP_Text outputText;
    [SerializeField] CameraShake cameraShake;
 
    IWorker worker;
    NetworkManager networkManager;
    Tron tron;
    PlayerInput playerInput;

    [HideInInspector] public GameState State;
    public float tickRate = 0.15f; //.15 seconds per tick
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
        var model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);

        networkManager = GetComponent<NetworkManager>();
        playerColor = Constants.cyan;
        adversaryColor = Constants.orange;
        tron = new Tron(new Vector2Int(25, 25));
    }

    public void Reset()
    {
        State = GameState.Playing;
        time = 0;

        tron.Reset();
        board.Reset();
        player.Reset(1, tron.bike1.pos);
        adversary.Reset(3, tron.bike2.pos);
        
        currentAction = 1;
        inputQueue.Clear();
        commited = false;

        from = tron.bike1.pos - DIRS[currentAction] * 0.5f;
        advFrom = tron.bike2.pos - DIRS[3] * 0.5f;
        advTo = advFrom + DIRS[3];
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
        int advAction = Adversary.ChooseMove(tron.trails, tron.bike2.pos, tron.bike1.pos);
        int playerAction = action;
        history.Add(new (playerAction, advAction));

        State = tron.Step(IDIRS[playerAction], IDIRS[advAction]);
        if (State != GameState.Playing) { EndEpisode(playerAction); }

        // Rendering and lerping setup
        player.Transform(playerAction, tron.bike1.pos);
        adversary.Transform(advAction, tron.bike2.pos);

        from = tron.bike1.pos - DIRS[currentAction] * 0.5f;
        advFrom = tron.bike2.pos - DIRS[advAction] * 0.5f;
        advTo = advFrom + DIRS[advAction];
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

    void RunInference()
    {
        Tensor input = new Tensor(1,11,11,3);
        // Fill input with random data
        for (int i = 0; i < input.length; i++)
        {
            input[i] = Random.Range(0f, 1f);
        }
        worker.Execute(input);
        Tensor output = worker.PeekOutput();
        // Debug.Log(output); // example
        outputText.text = output[0].ToString();
        input.Dispose();
        output.Dispose();
    }
}
