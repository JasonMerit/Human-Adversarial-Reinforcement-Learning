extends Node2D

@onready var player = $Player
@onready var adversary = $Adversary

const TICK_RATE = 1.0  # Seconds
const CELL_SIZE = 100

var tron: Tron

var time := 0.0

var actions = [Vector2i.UP, Vector2i.RIGHT, Vector2i.DOWN, Vector2i.LEFT]
var action_index = Vector2i.ZERO

func _ready():
	tron = Tron.new(10, 10)	
	reset()

func reset():
	tron.reset()
	tron.tick(Vector2i.RIGHT, Vector2i.LEFT)  # Start moving so trails are visible
	action_index = Vector2i.ZERO
	tron.bike1.last_pos = (tron.bike1.pos + Vector2i.LEFT) * CELL_SIZE
	tron.bike2.last_pos = (tron.bike2.pos + Vector2i.RIGHT) * CELL_SIZE

func _process(delta):
	if Input.is_action_just_pressed("ui_cancel"):
		get_tree().quit()

	time += delta
	if time >= TICK_RATE:
		time -= TICK_RATE

		tron.bike1.last_pos = tron.bike1.pos * CELL_SIZE
		tron.bike2.last_pos = tron.bike2.pos * CELL_SIZE

		action_index.x = (action_index.x + 1) % 4
		action_index.y = (action_index.y - 1) % 4
		var result = tron.tick(actions[1], actions[3])
		# var result = tron.tick(actions[action_index.x], actions[action_index.y])


		if result != 0:
			print("RESET")
			reset()

	# Interpolate render positions toward authoritative positions
	var alpha = time / TICK_RATE
	player.position = tron.bike1.last_pos.lerp(tron.bike1.pos * CELL_SIZE, alpha)
	adversary.position = tron.bike2.last_pos.lerp(tron.bike2.pos * CELL_SIZE, alpha)
