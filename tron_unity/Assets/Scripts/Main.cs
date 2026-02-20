using UnityEngine;
using Unity.Barracuda;
using TMPro;


public class Main : MonoBehaviour
{

    public NNModel modelAsset;
    public TMP_Text outputText;

    private IWorker worker;

    void Start()
    {
        var model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);
        
        RunInference();
        InvokeRepeating("RunInference", 1f, 1f);
    }

    void Update()
    {
        // Quit if ESC
        // if (Input.GetKeyDown(KeyCode.Escape))
        // {
        //     Application.Quit();
        // }
    }

    void RunInference()
    {

        Tensor input = new Tensor(1,11,11,3);
        // Fill input with random data
        for (int i = 0; i < input.length; i++)
        {
            input[i] = Random.Range(0f, 1f);
        }
        worker.Execute(input);
        Tensor output = worker.PeekOutput();
        Debug.Log(output); // example
        outputText.text = output[0].ToString();
        input.Dispose();
        output.Dispose();
    }
}
