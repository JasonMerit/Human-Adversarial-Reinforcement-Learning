using UnityEngine;
using System.Collections;
using System.Text;
using UnityEngine.Networking;
using System.Collections.Generic;

public class NetworkManager : MonoBehaviour
{
    private const string SUPABASE_PROJECT_REF = "bdjoehhrxfjumlphkbbg";
    private const string SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJkam9laGhyeGZqdW1scGhrYmJnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE2MDkwMDEsImV4cCI6MjA4NzE4NTAwMX0.PF2OQkcLTyc0pk_sy--T1jdhQxWaWD_DCw-xcLTHbkU";

    private const string EDGE_FUNCTION_URL =
        "https://" + SUPABASE_PROJECT_REF + ".supabase.co/functions/v1/upload-episode";

    public void SendEpisode(List<Vector2Int> trajectory, int winner, bool trapped)
    {
        StartCoroutine(SendEpisodeInternal(trajectory, winner, trapped));
    }

    public static string GetOrCreatePlayerToken()
    {
        string PLAYER_TOKEN_KEY = "playerToken";
        if (PlayerPrefs.HasKey(PLAYER_TOKEN_KEY)) return PlayerPrefs.GetString(PLAYER_TOKEN_KEY);
        else
        {
            string newToken = System.Guid.NewGuid().ToString("N").Substring(0, 12);
            PlayerPrefs.SetString(PLAYER_TOKEN_KEY, newToken);
            PlayerPrefs.Save(); // Ensure it's written immediately
            return newToken;
        }
    }

    private IEnumerator SendEpisodeInternal(List<Vector2Int> trajectory, int winner, bool trapped)

    {
        var payload = new EpisodePayload()
        {
            trajectory = trajectory,
            winner = winner,
            trapped = trapped,
        };

        var request = CreateRequest(EDGE_FUNCTION_URL, RequestType.POST, payload);

        AttachHeader(request, "Authorization", "Bearer " + SUPABASE_ANON_KEY);

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("Request failed: " + request.error);
            Debug.LogError("Response: " + request.downloadHandler.text);
        }
        else
        {
            Debug.Log("Response JSON: " + request.downloadHandler.text);
            var response = JsonUtility.FromJson<FunctionResponse>(
                request.downloadHandler.text
            );

            if (response != null && response.ok)
            {
                #if UNITY_EDITOR
                Debug.Log("Episode uploaded successfully.");
                #endif
            }
            else if (response != null)
            {
                Debug.LogError("Server error: " + response.error);
            }
        }
    }

    private UnityWebRequest CreateRequest(string path, RequestType type = RequestType.GET, object data = null)
    {
        var request = new UnityWebRequest(path, type.ToString());

        if (data != null)
        {
            var bodyRaw = Encoding.UTF8.GetBytes(JsonUtility.ToJson(data));
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        }

        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        return request;
    }

    private void AttachHeader(UnityWebRequest request, string key, string value)
    {
        request.SetRequestHeader(key, value);
    }
}

public enum RequestType
{
    GET = 0,
    POST = 1,
    PUT = 2
}

[System.Serializable]
public class EpisodePayload
{
    public List<Vector2Int> trajectory;
    public int winner;
    public bool trapped;
    public Vector3Int buildVersion = Main.BuildVersion;
    public string playerToken = NetworkManager.GetOrCreatePlayerToken();
}

[System.Serializable]
public class FunctionResponse
{
    public bool ok;
    public string error;
}