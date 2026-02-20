using UnityEngine;
using System.Collections;
using System.Text;
using UnityEngine.Networking;


public class NetworkManager : MonoBehaviour {
    private void Start() {
        StartCoroutine(MakeRequests());
    }

    private IEnumerator MakeRequests() {
        // GET
        // var getRequest = CreateRequest("https://jsonplaceholder.typicode.com/todos/1");
        // yield return getRequest.SendWebRequest();
        // Debug.Log("GET response JSON: " + getRequest.downloadHandler.text);
        // var deserializedGetData = JsonUtility.FromJson<Todo>(getRequest.downloadHandler.text);

        // POST
        var dataToPost = new PostData(){Hero = "John Wick", PowerLevel = 9001};
        var postRequest = CreateRequest("https://reqbin.com/echo/post/json", RequestType.POST, dataToPost);
        yield return postRequest.SendWebRequest();
        Debug.Log("POST response JSON: " + postRequest.downloadHandler.text);
        var deserializedPostData = JsonUtility.FromJson<PostResult>(postRequest.downloadHandler.text);

        // Trigger continuation of game flow
    }


    private UnityWebRequest CreateRequest(string path, RequestType type = RequestType.GET, object data = null) {
        var request = new UnityWebRequest(path, type.ToString());

        if (data != null) {
            var bodyRaw = Encoding.UTF8.GetBytes(JsonUtility.ToJson(data));
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        }

        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        return request;
    }

    private void AttachHeader(UnityWebRequest request,string key,string value)
    {
        request.SetRequestHeader(key, value);
    }
}

public enum RequestType {
    GET = 0,
    POST = 1,
    PUT = 2
}


public class Todo {
    // Ensure no getters / setters
    // Typecase has to match exactly
    public int userId;
    public int id;
    public string title;
    public bool completed;
}

[System.Serializable]
public class PostData {
    public string Hero;
    public int PowerLevel;
}

[System.Serializable]
public class PostResult
{
    public bool success;         // match type
    public PostData data;        // nested object if the JSON contains it
}