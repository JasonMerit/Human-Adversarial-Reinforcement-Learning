using UnityEngine;
using TMPro;
using UnityEngine.InputSystem;

public class Main : MonoBehaviour
{
    [SerializeField] TMP_Text centerText;
    Game game;

    enum GameState { WaitingToStart, Countdown, Playing, GameOver }
    GameState state = GameState.WaitingToStart;
    float countdownTime = 3f;
    float countdownTimer = 0f;

    void Start()
    {
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
            case GameState.WaitingToStart:
                centerText.text = "PRESS SPACE TO START";
                centerText.gameObject.SetActive(true);

                if (Keyboard.current.spaceKey.wasPressedThisFrame) { StartCountdown();}
                break;

            case GameState.Countdown:
                RunCountdown();
                break;

            case GameState.Playing:
                if (game.Tick()) { state = GameState.GameOver; }
                break;

            case GameState.GameOver:
                centerText.text = "GAME OVER\nPRESS SPACE";
                centerText.gameObject.SetActive(true);

                if (Keyboard.current.spaceKey.wasPressedThisFrame) { StartCountdown(); }
                break;
        }
    }

    void StartCountdown()
    {
        countdownTimer = 0f;
        game.Reset();
        state = GameState.Countdown;
        countdownTimer = countdownTime;
        centerText.gameObject.SetActive(true);
    }

    void RunCountdown()
    {
        countdownTimer -= Time.deltaTime;
        if (countdownTimer <= 0f)
        {
            state = GameState.Playing;
            centerText.gameObject.SetActive(false);
        }
        else
        {
            centerText.text = Mathf.Ceil(countdownTimer).ToString();
        }
    }
}
