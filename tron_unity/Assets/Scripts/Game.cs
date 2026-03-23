using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;
using UnityEngine.InputSystem;
using System.Linq;
using TMPro;
using System.Threading;


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
    Controller controller;
    PlayerInput playerInput;

    [HideInInspector] public GameState State;
    public float tickRate = 0.15f; //.15 seconds per tick
    List<Vector2Int> history = new();

    float time;
    [HideInInspector] public Color playerColor;
    [HideInInspector] public Color adversaryColor;

    void Awake()
    {
        controller = GetComponent<Controller>();
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
        time = tickRate; // immediate first tick
        tron.Reset();
        board.Reset();
        controller.Reset();
        player.Reset(1);
        adversary.Reset(3);
        currentAction = 1;
        inputQueue.Clear();

        lastPoint = new Vector2(tron.bike1.pos.x, tron.bike1.pos.y);
        targetPoint = lastPoint + DIRS[currentAction];

        player.transform.position = (Vector3)lastPoint;
        // player.transform.position = new Vector3(tron.bike1.pos.x, tron.bike1.pos.y, 0);
        adversary.transform.position = new Vector3(tron.bike2.pos.x, tron.bike2.pos.y, 0);

    }

    int currentAction = 1;
    Queue<int> inputQueue = new Queue<int>(2);

    Vector2 lastPoint;    // logical
    Vector2 targetPoint;  // next logical tile
    bool kek = false;

    public void Tick()
    {
        // player.AddTrail(player.transform.position, currentAction);  // Try moving down later
        
        //  Get input here
        int newAction = -1;
        if (playerInput.actions["Up"].triggered) newAction = 0;
        else if (playerInput.actions["Right"].triggered) newAction = 1;
        else if (playerInput.actions["Down"].triggered) newAction = 2;
        else if (playerInput.actions["Left"].triggered) newAction = 3;
        if (Keyboard.current.enterKey.wasPressedThisFrame) Time.timeScale = 1f;
        
        if (newAction != -1) { 
            currentAction = newAction; 
            targetPoint = lastPoint + DIRS[currentAction];
            player.Rotate(currentAction);
        }

        // Advance simulation by deltaTime
        time += Time.deltaTime;
        if (time >= tickRate)
        {
            time -= tickRate;
            Step(currentAction); // advance tile
            lastPoint = targetPoint;
            targetPoint = lastPoint + DIRS[currentAction];
        }

        // Interpolate for smooth visuals
        Vector2 entry = lastPoint - DIRS[currentAction] * 0.5f;
        Vector2 exit  = lastPoint + DIRS[currentAction] * 0.5f;

        float alpha = time / tickRate;
        if (newAction != -1) { 
            outputText.text = $"Alpha: {alpha:F2}";
        }
        
        player.transform.position = Vector2.Lerp(entry, exit, alpha);
        // player.transform.position = Vector2.Lerp(lastPoint, targetPoint, alpha);
        // player.transform.position = Vector2.Lerp(tron.bike1.lastPos, tron.bike1.pos, alpha);
        // adversary.transform.position = Vector2.Lerp(tron.bike2.lastPos, tron.bike2.pos, alpha);
        // adversary.AddTrail(adversary.transform.position);
    }

    void Step(int action)
    // Updates State
    {
        // board.SetCell(tron.bike1.pos, playerColor);
        // board.SetCell(tron.bike2.pos, adversaryColor);
        
        int advAction = Adversary.ChooseMove(tron.trails, tron.bike2.pos, tron.bike1.pos);
        // int playerAction = controller.GetAction();
        int playerAction = action;
        history.Add(new (playerAction, advAction));

        player.Transform(playerAction, tron.bike1.pos);
        // adversary.Rotate(advAction);
        
        State = tron.Step(IDIRS[playerAction], IDIRS[advAction]);
        if (State != GameState.Playing) { EndEpisode(playerAction); }
        
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
