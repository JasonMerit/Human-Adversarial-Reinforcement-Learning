using UnityEngine;


public class HTTP : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // Make a HTTP request to ping google
        WWW www = new WWW("http://www.google.com");
        while (!www.isDone)
        {
            // Wait for the request to complete
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
