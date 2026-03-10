using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;
using TMPro;


public class Game : MonoBehaviour
{
    public readonly Vector2Int[] DIRS = { Vector2Int.up, Vector2Int.right, Vector2Int.down, Vector2Int.left };

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

    [HideInInspector] public GameState State;
    float tickRate = 0.15f; // seconds per tick
    List<Vector2Int> history = new();

    float time;
    [HideInInspector] public Color playerColor;
    [HideInInspector] public Color adversaryColor;

    void Awake()
    {
        controller = GetComponent<Controller>();
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

        player.transform.position = new Vector3(tron.bike1.pos.x, tron.bike1.pos.y, 0);
        adversary.transform.position = new Vector3(tron.bike2.pos.x, tron.bike2.pos.y, 0);

    }

    public void Tick()
    {
        time += Time.deltaTime;
        if (time >= tickRate) {
            time -= tickRate;
            Step();
        }

        var alpha = time / tickRate;
        player.transform.position = (Vector3)Vector2.Lerp(tron.bike1.lastPos, tron.bike1.pos, alpha);
        adversary.transform.position = (Vector3)Vector2.Lerp(tron.bike2.lastPos, tron.bike2.pos, alpha);

        player.AddTrail(player.transform.position);
        adversary.AddTrail(adversary.transform.position);
    }

    void Step()
    // Updates State
    {
        // board.SetCell(tron.bike1.pos, playerColor);
        // board.SetCell(tron.bike2.pos, adversaryColor);
        
        int advAction = Adversary.ChooseMove(tron.trails, tron.bike2.pos, tron.bike1.pos);
        int playerAction = controller.GetAction(); 
        history.Add(new (playerAction, advAction));

        player.Rotate(playerAction);
        adversary.Rotate(advAction);
        
        State = tron.Step(DIRS[playerAction], DIRS[advAction]);
        if (State != GameState.Playing) { EndEpisode(playerAction); }
        
    }

    void EndEpisode(int playerAction)
    {
        if (State == GameState.Bike2Win) {
            cameraShake.Shake(DIRS[playerAction]);
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
        foreach (var dir in DIRS) { if (!bike.IsHitInDir(tron.trails, dir)) return false; }
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
