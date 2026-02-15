extends Node2D

@onready var player = $Player
@onready var adversary = $Adversary

var tron: Tron

func _ready():
	tron = Tron.new(10, 10)


func _process(delta):
	var result = tron.tick(Vector2i.RIGHT, Vector2i.LEFT)
	
	if result != 0:
		print("Game over:", result)
