using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;
using TMPro;
using UnityEngine.InputSystem;


public class Game : MonoBehaviour
{
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
    [SerializeField] new CameraShake camera;
 
    PlayerInput playerInput;
    IWorker worker;
    NetworkManager networkManager;
    Tron tron;

    [HideInInspector] public GameState State;
    const float tickRate = 0.2f; // seconds per tick
    List<Vector2Int> history = new();

    // Player
    int playerAction = 1;
    int lastPlayerAction = 1;

    float time;
    [HideInInspector] public Color playerColor;
    [HideInInspector] public Color adversaryColor;

    void Awake()
    {
        playerInput = GetComponent<PlayerInput>();
        var model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);

        networkManager = GetComponent<NetworkManager>();
        playerColor = player.GetComponent<SpriteRenderer>().color;
        adversaryColor = adversary.GetComponent<SpriteRenderer>().color;
        tron = new Tron(new Vector2Int(11, 11));
    }

    public void Reset()
    {
        time = tickRate; // immediate first tick
        tron.Reset();
        board.Clear();

        playerAction = 1;
        
        player.position = new Vector3(tron.bike1.pos.x, tron.bike1.pos.y, 0);
        adversary.position = new Vector3(tron.bike2.pos.x, tron.bike2.pos.y, 0);

    }

    public void Tick()
    {
        // Player input
        int newAction = playerAction;
        if (playerInput.actions["Up"].triggered) newAction = 0;
        else if (playerInput.actions["Right"].triggered) newAction = 1;
        else if (playerInput.actions["Down"].triggered) newAction = 2;
        else if (playerInput.actions["Left"].triggered) newAction = 3;
        if ((newAction + 2) % 4 != lastPlayerAction) playerAction = newAction; // prevent reversing
            

        time += Time.deltaTime;
        if (time >= tickRate) {
            time -= tickRate;
            Step();
        }

        var alpha = time / tickRate;
        player.position = (Vector3)Vector2.Lerp(tron.bike1.lastPos, tron.bike1.pos, alpha);
        adversary.position = (Vector3)Vector2.Lerp(tron.bike2.lastPos, tron.bike2.pos, alpha);
    }

    void Step()
    // Updates State
    {
        board.SetCell(tron.bike1.pos, playerColor);
        board.SetCell(tron.bike2.pos, adversaryColor);
        
        int advAction = Adversary.ChooseMove(tron.walls, tron.bike2.pos, tron.bike1.pos);
        history.Add(new (playerAction, advAction));
        lastPlayerAction = playerAction;

        State = tron.Step(DIRS[playerAction], DIRS[advAction]);
        if (State != GameState.Playing) { EndEpisode(); }
    }

    void EndEpisode()
    {
        camera.Shake();
        if (Main.PostingEnabled) { networkManager.SendEpisode(history, (int)State); }
        history = new();
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
