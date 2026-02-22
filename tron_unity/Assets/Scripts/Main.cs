using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;
using TMPro;
using UnityEngine.InputSystem;


public class Main : MonoBehaviour
{
    bool POSTING_ENABLED = false;

    readonly Vector2Int[] DIRS = 
    {
        new Vector2Int(0,1),   // Up
        new Vector2Int(1,0),   // Right
        new Vector2Int(0,-1),  // Down
        new Vector2Int(-1,0)   // Left
    };

    [SerializeField] Board board;
    [SerializeField] Transform player;
    [SerializeField] Transform adversary;
    [SerializeField] NNModel modelAsset;
    [SerializeField] TMP_Text outputText;
 
    PlayerInput playerInput;
    IWorker worker;
    NetworkManager networkManager;
    Tron tron;

    const float tickRate = 0.5f; // seconds per tick
    List<Vector2Int> trajectory = new List<Vector2Int>() {
        new(1, 3), new(1, 3), new(0, 2),
        new(3, 1), new(3, 1), new(3, 1), new(3, 1),
        new(2, 0), new(2, 0), new(1, 3),
        new(1, 3), new(1, 3), new(1, 3),
        new(1, 3), new(0, 2)
    };
    int step = 0;
    List<Vector2Int> history = new();

    // Player
    int playerAction = 1;

    float time;
    Color playerColor;
    Color adversaryColor;

    void Start()
    {
        playerInput = GetComponent<PlayerInput>();
        var model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);

        networkManager = GetComponent<NetworkManager>();

        // InvokeRepeating("RunInference", 0f, 1f);

        playerColor = player.GetComponent<SpriteRenderer>().color;
        adversaryColor = adversary.GetComponent<SpriteRenderer>().color;
        tron = new Tron(new Vector2Int(11, 11));
        Reset();
    }

    void Reset()
    {
        tron.Reset();
        board.Clear();

        board.SetCell(tron.bike1.pos, playerColor);
        board.SetCell(tron.bike2.pos, adversaryColor);
        tron.Tick(DIRS[1], DIRS[3]);
        
        player.position = new Vector3(tron.bike1.pos.x, tron.bike1.pos.y, 0);
        adversary.position = new Vector3(tron.bike2.pos.x, tron.bike2.pos.y, 0);
    }

    void Update()
    {
        // Input
        #if UNITY_EDITOR
        if (playerInput.actions["Quit"].triggered) { UnityEditor.EditorApplication.isPlaying = false; }
        #endif

        int newAction = playerAction;
        if (playerInput.actions["Up"].triggered) newAction = 0;
        else if (playerInput.actions["Right"].triggered) newAction = 1;
        else if (playerInput.actions["Down"].triggered) newAction = 2;
        else if (playerInput.actions["Left"].triggered) newAction = 3;
        if ((newAction + 2) % 4 != playerAction) playerAction = newAction; // prevent reversing
            

        time += Time.deltaTime;
        if (time >= tickRate) {
            time -= tickRate;
            Tick();
        }

        var alpha = time / tickRate;
        player.position = (Vector3)Vector2.Lerp(tron.bike1.lastPos, tron.bike1.pos, alpha);
        adversary.position = (Vector3)Vector2.Lerp(tron.bike2.lastPos, tron.bike2.pos, alpha);

    }

    void Tick()
    {
        board.SetCell(tron.bike1.pos, playerColor);
        board.SetCell(tron.bike2.pos, adversaryColor);
        Vector2Int action = trajectory[step];
        step = (step + 1) % trajectory.Count;

        history.Add(action);
        
        Vector2Int dir1 = DIRS[action.x];
        Vector2Int dir2 = DIRS[action.y];
        
        // Vector2Int dir1 = DIRS[1];
        // Vector2Int dir2 = DIRS[3];

        Vector2Int playerDir = DIRS[playerAction];
        Vector2Int advDir = Adversary.ChooseMove(tron.walls, tron.bike1.pos, tron.bike2.pos);

        dir1 = playerDir;
        dir2 = advDir;

        int result = tron.Tick(dir1, dir2);
        // Debug.Log("Tick result: " + result);

        if (result != -1)
        {
            if (POSTING_ENABLED) { networkManager.SendEpisode(history, result); }
            history = new List<Vector2Int>();
            Reset();
            step = 0;
        }
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
