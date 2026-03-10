using UnityEngine;
using UnityEngine.InputSystem;

public class Controller : MonoBehaviour
{
    readonly Vector2Int[] DIRS = { Vector2Int.up, Vector2Int.right, Vector2Int.down, Vector2Int.left };

    PlayerInput playerInput;
    int action = 1;
    int lastAction = 1;

    void Awake()
    {
        playerInput = GetComponent<PlayerInput>();
    }

    void Update()
    {
        int newAction = action;
        if (playerInput.actions["Up"].triggered) newAction = 0;
        else if (playerInput.actions["Right"].triggered) newAction = 1;
        else if (playerInput.actions["Down"].triggered) newAction = 2;
        else if (playerInput.actions["Left"].triggered) newAction = 3;
        if ((newAction + 2) % 4 != lastAction) action = newAction; // prevent reversing
    }

    // Called every Game.Step
    public int GetAction() 
    {
        lastAction = action;
        return action;
    }
}