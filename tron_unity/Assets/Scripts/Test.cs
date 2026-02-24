using UnityEngine;
using TMPro;
using UnityEngine.InputSystem;
using System.Collections.Generic;

public class Test : MonoBehaviour
{
    public readonly Vector2Int[] DIRS = 
    {
        new(0,1),   // Up
        new(1,0),   // Right
        new(0,-1),  // Down
        new(-1,0)   // Left
    };

    [SerializeField] Board board;
    [SerializeField] Board boardGhost;
    [SerializeField] Transform player;
    [SerializeField] Transform adversary;
    [SerializeField] TMP_Text outputText;
 
    Tron tron;

    [HideInInspector] public Color playerColor;
    [HideInInspector] public Color adversaryColor;

    List<Vector2Int> actionHistory = new();
    List<float> scoreHistory = new();

    void Start()
    {
        playerColor = player.GetComponent<SpriteRenderer>().color;
        adversaryColor = adversary.GetComponent<SpriteRenderer>().color;
        tron = new Tron(new Vector2Int(11, 11));
        Reset();
    }

    public void Reset()
    {
        tron.Reset();
        board.Clear();
        boardGhost.Clear();
        player.position = new Vector3(tron.bike1.pos.x, tron.bike1.pos.y, 0);
        adversary.position = new Vector3(tron.bike2.pos.x, tron.bike2.pos.y, 0);

    }

    Vector2Int action = new(-1,-1);
    void Update()
    {
        #if UNITY_EDITOR
        if (Keyboard.current.escapeKey.wasPressedThisFrame) { UnityEditor.EditorApplication.isPlaying = false; }
        if (Keyboard.current.rKey.wasPressedThisFrame) { Reset(); }
        #endif

        Vector2Int newAction = action;
        // Player input
        if (Keyboard.current.wKey.wasPressedThisFrame) newAction.x = 0;
        else if (Keyboard.current.dKey.wasPressedThisFrame) newAction.x = 1;
        else if (Keyboard.current.sKey.wasPressedThisFrame) newAction.x = 2;
        else if (Keyboard.current.aKey.wasPressedThisFrame) newAction.x = 3;

        // aAdv input
        if (Keyboard.current.upArrowKey.wasPressedThisFrame) newAction.y = 0;
        else if (Keyboard.current.rightArrowKey.wasPressedThisFrame) newAction.y = 1;
        else if (Keyboard.current.downArrowKey.wasPressedThisFrame) newAction.y = 2;
        else if (Keyboard.current.leftArrowKey.wasPressedThisFrame) newAction.y = 3;

        // print histories upon "P" key press
        if (Keyboard.current.pKey.wasPressedThisFrame) {
            string log = "[";
            for (int i = 0; i < actionHistory.Count; i++) { log += $"({actionHistory[i].x}, {actionHistory[i].y}), "; }
            log += "]\n[";
            for (int i = 0; i < scoreHistory.Count; i++) { log += $"{scoreHistory[i]}, "; }
            log += "]";
            Debug.Log(log);
        }
        
        if (newAction.x > -1 && !tron.bike1.IsHitInDir(tron.trails, DIRS[newAction.x])) action.x = newAction.x;
        if (newAction.y > -1 && !tron.bike2.IsHitInDir(tron.trails, DIRS[newAction.y])) action.y = newAction.y;

        if (action.x != -1 && action.y != -1) { 
            Step();
            int score = Heuristic.Voronoi(tron.trails, tron.bike1.pos, tron.bike2.pos);
            int[,] ownership = Heuristic.FillBoard(tron.trails, tron.bike1.pos, tron.bike2.pos);
            int playerScore = 0;
            int adversaryScore = 0;
            boardGhost.Clear();
            for (int x = 0; x < ownership.GetLength(0); x++) {
                for (int y = 0; y < ownership.GetLength(1); y++) {
                    if (tron.trails[x, y] != 0) continue; // skip trails

                    Color color;
                    if (ownership[x, y] == 1) {
                        color = playerColor;
                        playerScore++;
                    }
                    else if (ownership[x, y] == 2) {
                        color = adversaryColor;
                        adversaryScore++;
                    }
                    else if (ownership[x, y] == 3) color = Color.yellow; // battlefront
                    else color = Color.gray;

                    color.a = 0.5f;
                    boardGhost.SetCell(new Vector2Int(x, y), color);
                }
            }
            // outputText.text = "Player: " + playerScore + "\nAdversary: " + adversaryScore;
            outputText.text = "Score: " + score;
            scoreHistory.Add(score);
         }
    }

    void Step()
    {
        board.SetCell(tron.bike1.pos, playerColor);
        board.SetCell(tron.bike2.pos, adversaryColor);

        GameState state = tron.Step(DIRS[action.x], DIRS[action.y]);
        // Add to history, etc.
        actionHistory.Add(action);
        action = new(-1,-1);

        player.position = new Vector3(tron.bike1.pos.x, tron.bike1.pos.y, 0);
        adversary.position = new Vector3(tron.bike2.pos.x, tron.bike2.pos.y, 0);

        if (state != GameState.Playing) {
            Debug.Log("Game Over: " + state);
        }
    }

}
