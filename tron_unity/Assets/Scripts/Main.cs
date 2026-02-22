using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;
using TMPro;
using UnityEngine.InputSystem;


public class Main : MonoBehaviour
{
    bool POSTING_ENABLED = false;

    public readonly Vector2Int[] DIRS = 
    {
        new(0,1),   // Up
        new(1,0),   // Right
        new(0,-1),  // Down
        new(-1,0)   // Left
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

    const float tickRate = 0.2f; // seconds per tick
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
        time = 0;
        tron.Reset();
        board.Clear();

        board.SetCell(tron.bike1.pos, playerColor);
        board.SetCell(tron.bike2.pos, adversaryColor);
        playerAction = 1;
        tron.Tick(DIRS[playerAction], DIRS[3]);
        
        player.position = new Vector3(tron.bike1.pos.x, tron.bike1.pos.y, 0);
        adversary.position = new Vector3(tron.bike2.pos.x, tron.bike2.pos.y, 0);

    }

    void Update()
    {
        // Input
        #if UNITY_EDITOR
        if (playerInput.actions["Quit"].triggered) { UnityEditor.EditorApplication.isPlaying = false; }
        if (playerInput.actions["Restart"].triggered) { Reset(); return;}
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
        
        int advAction = Adversary.ChooseMove(tron.walls, tron.bike2.pos, tron.bike1.pos);
        history.Add(new (playerAction, advAction));

        int result = tron.Tick(DIRS[playerAction], DIRS[advAction]);
        // Debug.Log("Tick result: " + result);

        if (result != -1)
        {
            if (POSTING_ENABLED) { networkManager.SendEpisode(history, result); }
            history = new List<Vector2Int>();
            Reset();
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
