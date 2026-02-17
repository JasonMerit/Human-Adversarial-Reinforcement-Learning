extends Node2D

@onready var player = $Player
@onready var adversary = $Adversary
@onready var uploader = $TrajectoryUploader

const TICK_RATE = .2  # Seconds
const CELL_SIZE = 100
const GRID_SIZE = Vector2i(11, 11)

var tron: Tron

var time := 0.0

const actions = [Vector2i.UP, Vector2i.RIGHT, Vector2i.DOWN, Vector2i.LEFT]
var action_index = Vector2i.ZERO

var draw_walls: Array[Vector2i]

# HACK
const trajectory: Array[Vector2i] = [
	Vector2i(1, 3), Vector2i(1, 3), Vector2i(0, 2), 
	Vector2i(3, 1), Vector2i(3, 1), Vector2i(3, 1), Vector2i(3, 1),
	Vector2i(2, 0), Vector2i(2, 0), Vector2i(1, 3),
	Vector2i(1, 3), Vector2i(1, 3), Vector2i(1, 3), Vector2i(1, 3), Vector2i(0, 2)]
var trajectory_index = 0
var history: Array[Vector2i] = []

func _ready():
	tron = Tron.new(GRID_SIZE)	
	reset()

func reset():
	tron.reset()
	draw_walls = [tron.bike1.pos, tron.bike2.pos]
	tron.tick(Vector2i.RIGHT, Vector2i.LEFT)

	action_index = Vector2i.ZERO
	trajectory_index = 0
	tron.bike1.last_pos = (tron.bike1.pos + Vector2i.LEFT) * CELL_SIZE
	tron.bike2.last_pos = (tron.bike2.pos + Vector2i.RIGHT) * CELL_SIZE
	queue_redraw()

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


func tick():
	tron.bike1.last_pos = tron.bike1.pos * CELL_SIZE
	tron.bike2.last_pos = tron.bike2.pos * CELL_SIZE
	draw_walls += [tron.bike1.pos, tron.bike2.pos]

	# action_index.x = (action_index.x + 1) % 4
	# action_index.y = (action_index.y - 1) % 4
	# var result = tron.tick(actions[action_index.x], actions[action_index.y])

	var action = trajectory[trajectory_index]
	var dir1 = actions[action.x]
	var dir2 = actions[action.y]
	trajectory_index = (trajectory_index + 1) % trajectory.size()

	var result = tron.tick(dir1, dir2)
	history.append(action)

	if result != -1:
		uploader.enqueue_trajectory(history.duplicate(), result)
		history.clear()

		# get_tree().quit()
		reset()

	queue_redraw()

func _draw() -> void:
	for wall in draw_walls:
		var color = player.modulate if tron.walls[wall.y][wall.x] == 1 else adversary.modulate
		draw_rect(Rect2(wall.x * CELL_SIZE, wall.y * CELL_SIZE, CELL_SIZE, CELL_SIZE), color)
		draw_rect(Rect2(wall.x * CELL_SIZE, wall.y * CELL_SIZE, CELL_SIZE, CELL_SIZE), Color.BLACK, false, 5.0)
	
