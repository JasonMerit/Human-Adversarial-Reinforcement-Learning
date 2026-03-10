using UnityEngine;
using System.Collections.Generic;
using UnityEngine.InputSystem;
using System.Linq;

public class Controller : MonoBehaviour
{
    readonly Vector2Int[] DIRS = { Vector2Int.up, Vector2Int.right, Vector2Int.down, Vector2Int.left };

    PlayerInput playerInput;
    int currentAction = 1;

    Queue<int> inputQueue = new Queue<int>(2);

    void Awake()
    {
        playerInput = GetComponent<PlayerInput>();
    }

    public void Reset()
    {
        currentAction = 1;
        inputQueue.Clear();
    }

    void Update()
    {
        if (playerInput.actions["Up"].triggered) RegisterInput(0);
        else if (playerInput.actions["Right"].triggered) RegisterInput(1);
        else if (playerInput.actions["Down"].triggered) RegisterInput(2);
        else if (playerInput.actions["Left"].triggered) RegisterInput(3);
    }

    void RegisterInput(int newAction)
    {
        if (inputQueue.Count >= 2)
            return;

        int last = inputQueue.Count > 0 ? inputQueue.Last() : currentAction;
        if (newAction == last || (newAction + 2) % 4 == last)
            return; // prevent duplicate and reverse

        inputQueue.Enqueue(newAction);
    }

    // Called every Game.Step
    public int GetAction() 
    {
        if (inputQueue.Count > 0)
            currentAction = inputQueue.Dequeue();

        return currentAction;
    }
}