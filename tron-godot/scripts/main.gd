extends Node2D

@onready var player = $Player
@onready var adversary = $Adversary

const TICK_RATE = .5  # Seconds
const CELL_SIZE = 100

var tron: Tron

var time := 0.0

var actions = [Vector2i.UP, Vector2i.RIGHT, Vector2i.DOWN, Vector2i.LEFT]
var action_index = Vector2i.ZERO

var draw_walls: Array[Vector2i]

func _ready():
	# Set viewport clearmode
	# RenderingServer.viewport_set_clear_mode(get_viewport().get_viewport_rid(), RenderingServer.VIEWPORT_CLEAR_NEVER)
	tron = Tron.new(10, 10)	
	reset()

func reset():
	tron.reset()
	draw_walls = [tron.bike1.pos, tron.bike2.pos]
	queue_redraw()
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
		tick()

	# Interpolate render positions toward authoritative positions
	var alpha = time / TICK_RATE
	player.position = tron.bike1.last_pos.lerp(tron.bike1.pos * CELL_SIZE, alpha)
	adversary.position = tron.bike2.last_pos.lerp(tron.bike2.pos * CELL_SIZE, alpha)

	# Queue the draw call to render the walls
	queue_redraw()

func tick():
	tron.bike1.last_pos = tron.bike1.pos * CELL_SIZE
	tron.bike2.last_pos = tron.bike2.pos * CELL_SIZE
	draw_walls += [tron.bike1.pos, tron.bike2.pos]
	queue_redraw()

	action_index.x = (action_index.x + 1) % 4
	action_index.y = (action_index.y - 1) % 4
	# var result = tron.tick(actions[1], actions[3])
	var result = tron.tick(actions[action_index.x], actions[action_index.y])


	if result != 0:
		print("RESET")
		reset()

func _draw() -> void:
	for wall in draw_walls:
		assert (tron.walls[wall.y][wall.x] != 0)
		var color = player.modulate if tron.walls[wall.y][wall.x] == 1 else adversary.modulate
		draw_rect(Rect2(wall.x * CELL_SIZE, wall.y * CELL_SIZE, CELL_SIZE, CELL_SIZE), color)
		draw_rect(Rect2(wall.x * CELL_SIZE, wall.y * CELL_SIZE, CELL_SIZE, CELL_SIZE), Color.BLACK, false, 5.0)
	
