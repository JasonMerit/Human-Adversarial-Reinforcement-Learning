using UnityEngine;
using TMPro;
using UnityEngine.InputSystem;

public class Main : MonoBehaviour
{
    public static readonly Vector3Int BuildVersion = new(1, 0, 0);
    public static readonly bool PostingEnabled = false;

    [SerializeField] TMP_Text versionText;
    [SerializeField] TMP_Text centerText;
    Game game;

    enum State { WaitingToStart, Countdown, Playing, GameOver }
    State state = State.WaitingToStart;
    float countdownTime = 3f;
    float countdownTimer = 0f;

    void Start()
    {
        versionText.text = $"{BuildVersion.x}.{BuildVersion.y}.{BuildVersion.z}";
        game = GetComponentInChildren<Game>();
        #if UNITY_EDITOR
        countdownTime = 0f;
        #endif
    }

    void Update()
    {
        // Input
        #if UNITY_EDITOR
        if (Keyboard.current.escapeKey.wasPressedThisFrame) { UnityEditor.EditorApplication.isPlaying = false; }
        if (Keyboard.current.rKey.wasPressedThisFrame) { StartCountdown(); }
        #endif

        switch (state)
        {
            case State.WaitingToStart:
                centerText.text = "PRESS SPACE TO START";
                centerText.gameObject.SetActive(true);

                if (Keyboard.current.spaceKey.wasPressedThisFrame) { StartCountdown();}
                break;

            case State.Countdown:
                RunCountdown();
                break;

            case State.Playing:
                game.Tick();
                if (game.State != GameState.Playing) { AnnounceWinner(game.State); }
                break;

            case State.GameOver:
                if (Keyboard.current.spaceKey.wasPressedThisFrame) { StartCountdown(); }
                break;
        }
    }

    void StartCountdown()
    {
        countdownTimer = 0f;
        game.Reset();
        state = State.Countdown;
        countdownTimer = countdownTime;
        centerText.gameObject.SetActive(true);
    }

    void RunCountdown()
    {
        countdownTimer -= Time.deltaTime;
        if (countdownTimer <= 0f)
        {
            state = State.Playing;
            centerText.gameObject.SetActive(false);
        }
        else
        {
            centerText.text = Mathf.Ceil(countdownTimer).ToString();
        }
    }

    void AnnounceWinner(GameState result)
    {
        if (result == GameState.Bike1Win) { 
            centerText.text = "YOU WIN!";
            centerText.color = game.playerColor;
        }
        else if (result == GameState.Bike2Win) { 
            centerText.text = "YOU LOSE!";
            centerText.color = game.adversaryColor;
        }
        else { 
            centerText.text = "DRAW!"; 
            centerText.color = Color.cyan;
        }
        
        // Color
        centerText.text += "\nPRESS SPACE";
        centerText.gameObject.SetActive(true);

        state = State.GameOver;
    }
}
