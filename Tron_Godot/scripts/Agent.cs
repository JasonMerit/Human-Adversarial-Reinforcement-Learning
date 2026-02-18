using Godot;
using System;
using System.Linq;
// using Microsoft.ML.OnnxRuntime;
// using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;

public class Agent : Sprite
{
    // private InferenceSession session;

    public override void _Ready()
    {
        GD.Print("Agent ready.");
        // try
        // {
        //     string path = ProjectSettings.GlobalizePath("res://agent.onnx");
        //     session = new InferenceSession(path);
        //     GD.Print("ONNX session created.");
        // }
        // catch (Exception e)
        // {
        //     GD.PrintErr(e.ToString());
        // }
        // session = new InferenceSession(ProjectSettings.GlobalizePath("res://agent.onnx"));
    }

    public void Act(Vector2 playerPosition, Vector2 enemyPosition)
    {
        
    }

}
