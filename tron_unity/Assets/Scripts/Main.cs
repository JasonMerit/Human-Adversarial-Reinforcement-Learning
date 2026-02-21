using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;
using TMPro;
using UnityEngine.InputSystem;


public class Main : MonoBehaviour
{
    List<Vector2Int> DIRS = new List<Vector2Int>()
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

    float time;
    Color playerColor;
    Color adversaryColor;

    void Start()
    {
        playerInput = GetComponent<PlayerInput>();
        var model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);

        networkManager = GetComponent<NetworkManager>();
        List<Vector2Int> exampleTrajectory = new List<Vector2Int>()
        {
            new Vector2Int(3,1),
            new Vector2Int(2,2),
            new Vector2Int(1,0)
        };

        // networkManager.SendEpisode(exampleTrajectory, 1);
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
        #if UNITY_EDITOR
        if (playerInput.actions["Quit"].triggered)
        {
            UnityEditor.EditorApplication.isPlaying = false;
        }
        #endif

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
        // Example: random moves for both bikes
        Vector2Int dir1 = DIRS[1];
        Vector2Int dir2 = DIRS[3];

        int result = tron.Tick(dir1, dir2);
        // Debug.Log("Tick result: " + result);

        if (result != -1)
        {
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
        Debug.Log(output); // example
        outputText.text = output[0].ToString();
        input.Dispose();
        output.Dispose();
    }
}
