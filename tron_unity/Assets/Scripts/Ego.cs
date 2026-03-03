using UnityEngine;
using TMPro;
using UnityEngine.InputSystem;
using System.Collections.Generic;
using UnityEngine.UIElements;
using System.Text.RegularExpressions;

public class Ego : MonoBehaviour
{
    public readonly Vector2Int[] DIRS = 
    {
        new(0,1),   // Up
        new(1,0),   // Right
        new(0,-1),  // Down
        new(-1,0)   // Left
    };

    [SerializeField] Board board;
    [SerializeField] TMP_Text outputText;
 
    Tron tron;

    [HideInInspector] public Color playerColor;
    [HideInInspector] public Color adversaryColor;

    int orientation = 3;

    void Start()
    {
        playerColor = new Color(0.1f, 0.8f, 0.1f);
        adversaryColor = new Color(0.8f, 0.1f, 0.1f);
        tron = new Tron(new Vector2Int(11, 11));
        Reset();
    }

    public void Reset()
    {
        tron.Reset();
        board.Clear();
        orientation = 3;
        DrawBoard();

    }

    void Update()
    {
        #if UNITY_EDITOR
        if (Keyboard.current.escapeKey.wasPressedThisFrame) { UnityEditor.EditorApplication.isPlaying = false; }
        if (Keyboard.current.rKey.wasPressedThisFrame) { Reset(); }
        #endif

        int action = -1;
        // Adversary input
        if (Keyboard.current.aKey.wasPressedThisFrame) action = 0;  // Turn left
        else if (Keyboard.current.wKey.wasPressedThisFrame) action = 1;  // Go straight
        else if (Keyboard.current.dKey.wasPressedThisFrame) action = 2;  // Turn right
        if (action > -1) {
            int new_orientation = (orientation + action + 4 - 1) % 4;
            if (!tron.bike2.IsHitInDir(tron.trails, DIRS[new_orientation])) {
                orientation = new_orientation;
                Step();
            }
        } 
    }

    void Step()
    {
        int humanAction = Adversary.ChooseMove(tron.trails, tron.bike1.pos, tron.bike2.pos);

        GameState state = tron.Step(DIRS[humanAction], DIRS[orientation]);

        if (state != GameState.Playing) {
            Reset();
            Debug.Log("Game Over: " + state);
        }
        DrawBoard();
    }

    void DrawBoard()
    {
        // Make a copy of tron.trails and rotate it 90 degrees clockwise for correct orientation on the tilemap
        board.Clear();
        int[,]  rotatedTrails = RotateMatrix(tron.trails, orientation);
        // int[,]  rotatedTrails = tron.trails;
        for (int x = 0; x < tron.width; x++)
        {
            for (int y = 0; y < tron.height; y++)
            {
                if (rotatedTrails[x, y] == 1) board.SetCell(new Vector2Int(x, y), playerColor);
                else if (rotatedTrails[x, y] == 2) board.SetCell(new Vector2Int(x, y), adversaryColor);
            }
        }
    }

    int[,] RotateMatrix(int[,] matrix, int orientation) {
        if (orientation == 0) return matrix; // No rotation needed for Left orientation
        int width = matrix.GetLength(0);
        int height = matrix.GetLength(1);
        int[,] rotated = new int[height, width];

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                switch (orientation) {
                    case 1: // Right
                        rotated[height - 1 - y, x] = matrix[x, y];
                        break;
                    case 2: // Down
                        rotated[height - 1 - x, width - 1 - y] = matrix[x, y];
                        break;
                    case 3: // Left
                        rotated[y, width - 1 - x] = matrix[x, y];
                        break;
                }
            }
        }
        return rotated;
    }
}
