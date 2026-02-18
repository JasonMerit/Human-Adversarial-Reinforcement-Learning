using Godot;
using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;

public class Agent : Sprite
{
    private InferenceSession session;

    public override void _Ready()
    {
        session = new InferenceSession("res://agent.onnx");
    }

    public float[] Run(float[,,] input)
    {
        // Input: [C,H,W], flatten to 1xC,H,W
        int c = input.GetLength(0);
        int h = input.GetLength(1);
        int w = input.GetLength(2);

        var tensor = new DenseTensor<float>(new int[] {1, c, h, w});
        for (int i = 0; i < c; i++)
            for (int j = 0; j < h; j++)
                for (int k = 0; k < w; k++)
                    tensor[0, i, j, k] = input[i, j, k];

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", tensor)
        };

        var results = session.Run(inputs);

        return results.First().AsEnumerable<float>().ToArray();
    }
}
