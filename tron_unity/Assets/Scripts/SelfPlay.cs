using UnityEngine;
using Unity.Barracuda;
using TMPro;
using NUnit.Framework;

public class SelfPlay : MonoBehaviour
{
    public readonly Vector2[] DIRS = { Vector2Int.up, Vector2Int.right, Vector2Int.down, Vector2Int.left };
    public readonly Vector2Int[] IDIRS = { Vector2Int.up, Vector2Int.right, Vector2Int.down, Vector2Int.left };

    [SerializeField] Board board;
    [SerializeField] Bike player;
    [SerializeField] Bike adversary;
    [SerializeField] NNModel humanModelAsset;
    [SerializeField] NNModel adversaryModelAsset;
    [SerializeField] TMP_Text outputText;
    [SerializeField] CameraShake cameraShake;
 
    IWorker humanWorker;
    IWorker adversaryWorker;
    Tron tron;

    [HideInInspector] public GameState State;
    public float tickRate = 0.15f; //.15 seconds per tick

    float time;
    [HideInInspector] public Color playerColor;
    [HideInInspector] public Color adversaryColor;

    // Input / action / lerping
    Vector2 from;
    Vector2 to;
    Vector2 advFrom;
    Vector2 advTo;

    void Awake()
    {
        // humanWorker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, ModelLoader.Load(humanModelAsset));
        humanWorker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpBurst, ModelLoader.Load(humanModelAsset));
        adversaryWorker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpBurst, ModelLoader.Load(adversaryModelAsset));

        playerColor = Constants.cyan;
        adversaryColor = Constants.orange;
        tron = new Tron(new Vector2Int(25, 25));
    }

    void Start() {
        Reset();
    }

    public void Reset()
    {
        State = GameState.Playing;
        time = 0;

        tron.Reset();
        board.Reset();
        player.Reset(1, tron.bike1.pos);
        adversary.Reset(3, tron.bike2.pos);
        

        from = tron.bike1.pos - DIRS[1] * 0.5f;
        to = from + DIRS[1];
        advFrom = tron.bike2.pos - DIRS[3] * 0.5f;
        advTo = advFrom + DIRS[3];
    }

    public void Update()
    {       
        time += Time.deltaTime;
        if (time >= tickRate)
        {
            time -= tickRate;
            Step(); // advance tile
        }

        // 5. Interpolate for smooth visuals and border based end points
        float alpha = time / tickRate;
        player.LerpPosition(from, to, alpha);
        adversary.LerpPosition(advFrom, advTo, alpha);
    }

    void Step()
    {
        int humanAction = GetAction(humanWorker, player.orientation, tron.trails, tron.bike1.pos, tron.bike2.pos);
        // Assert.IsTrue(humanAction == GetAction(humanWorker, player.orientation, tron.trails, tron.bike1.pos, tron.bike2.pos), $"nondeterministic human model");
        int humanHeading = (player.orientation + (humanAction - 1) + 4) % 4;

        int advAction = GetAction(adversaryWorker, adversary.orientation, tron.trails, tron.bike2.pos, tron.bike1.pos);
        // Assert.IsTrue(advAction == GetAction(adversaryWorker, adversary.orientation, tron.trails, tron.bike2.pos, tron.bike1.pos), $"nondeterministic adversary model");
        int advHeading = (adversary.orientation + (advAction - 1) + 4) % 4;

        State = tron.Step(IDIRS[humanHeading], IDIRS[advHeading]);
        if (State != GameState.Playing) { EndEpisode(humanHeading); return;}

        // Rendering and lerping setup
        player.Transform(humanHeading, tron.bike1.pos);
        adversary.Transform(advHeading, tron.bike2.pos);

        from = tron.bike1.pos - DIRS[humanHeading] * 0.5f;
        to = from + DIRS[humanHeading];
        advFrom = tron.bike2.pos - DIRS[advHeading] * 0.5f;
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

        Reset();
    }

    (int tx, int ty) Rotate(int x, int y, int size, int o)
    {
        if (o == 0) return (x, y);  // Up
        if (o == 1) return (size - 1 - y, x);  // Right
        if (o == 2) return (size - 1 - x, size - 1 - y);  // Down
        return (y, size - 1 - x);  // Left
    }

    int GetAction(IWorker worker, int orientation, int[,] trails, Vector2 you, Vector2 other)
    {
        // Debug.Log(orientation);
        // UnityEditor.EditorApplication.isPlaying = false;
        // Create tensor in a using block to ensure proper disposal
        using (var input = new Tensor(1, 25, 25, 3))
        {
            // Fill input with trails and bike positions
            for (int x = 0; x < 25; x++) {
                for (int y = 0; y < 25; y++) {
                    var (tx, ty) = Rotate(x, y, 25, orientation);

                    input[0, ty, tx, 0] = (trails[x, y] != 0) ? 1f : 0f;
                    input[0, ty, tx, 1] = (you.x == x && you.y == y) ? 1f : 0f;
                    input[0, ty, tx, 2] = (other.x == x && other.y == y) ? 1f : 0f;
                }
            }

            worker.Execute(input);

            // Peek output in another using block
            using (var output = worker.PeekOutput("q_values"))
            {
                int nActions = output.channels;
                float maxQ = float.NegativeInfinity;
                int bestAction = 0;

                for (int i = 0; i < nActions; i++)
                {
                    float qValue = output[0, 0, 0, i];
                    if (qValue > maxQ)
                    {
                        maxQ = qValue;
                        bestAction = i;
                    }
                }
                // Debug.Log($"Adversary chose action {bestAction} with Q-value {maxQ}");
                return bestAction;
            }
        }
    }

}
