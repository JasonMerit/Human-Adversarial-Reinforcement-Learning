extends Node2D

var Tron = preload("res://scripts/Tron.gd")

onready var player = $Player
onready var adversary = $Adversary
onready var uploader = $TrajectoryUploader

const TICK_RATE = 0.2
const CELL_SIZE = 50
const GRID_SIZE = Vector2(10, 11)

var tron = null
var time = 0.0

const actions = [Vector2.UP, Vector2.RIGHT, Vector2.DOWN, Vector2.LEFT]
var action_index = Vector2.ZERO

var draw_walls = []

# HACK
const trajectory = [
	Vector2(1, 3), Vector2(1, 3), Vector2(0, 2),
	Vector2(3, 1), Vector2(3, 1), Vector2(3, 1), Vector2(3, 1),
	Vector2(2, 0), Vector2(2, 0), Vector2(1, 3),
	Vector2(1, 3), Vector2(1, 3), Vector2(1, 3),
	Vector2(1, 3), Vector2(0, 2)
]

var trajectory_index = 0
var history = []

func _ready():
	tron = Tron.new(GRID_SIZE)
	reset()

func reset():
	tron.reset()

	draw_walls = [tron.bike1.pos, tron.bike2.pos]
	tron.tick(Vector2.RIGHT, Vector2.LEFT)

	action_index = Vector2.ZERO
	trajectory_index = 0

	tron.bike1.last_pos = (tron.bike1.pos + Vector2.LEFT) * CELL_SIZE
	tron.bike2.last_pos = (tron.bike2.pos + Vector2.RIGHT) * CELL_SIZE

	update()

func _process(delta):
	if Input.is_action_just_pressed("ui_cancel"):
		get_tree().quit()

	time += delta

	if time >= TICK_RATE:
		time -= TICK_RATE
		tick()

	var alpha = time / TICK_RATE
	player.position = tron.bike1.last_pos.linear_interpolate(tron.bike1.pos * CELL_SIZE, alpha)
	adversary.position = tron.bike2.last_pos.linear_interpolate(tron.bike2.pos * CELL_SIZE, alpha)

func tick():
	tron.bike1.last_pos = tron.bike1.pos * CELL_SIZE
	tron.bike2.last_pos = tron.bike2.pos * CELL_SIZE

	draw_walls += [tron.bike1.pos, tron.bike2.pos]

	var action = trajectory[trajectory_index]
	var dir1 = actions[int(action.x)]
	var dir2 = actions[int(action.y)]

	trajectory_index = (trajectory_index + 1) % trajectory.size()

	var result = tron.tick(dir1, dir2)
	history.append(action)

	if result != -1:
		# uploader.enqueue_trajectory(history, result)
		history = []
		reset()

	update()

func _draw():
	for wall in draw_walls:
		var color = player.modulate if tron.walls[wall.y][wall.x] == 1 else adversary.modulate
		var rect = Rect2(wall.x * CELL_SIZE, wall.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

		draw_rect(rect, color)
		draw_rect(rect, Color.black, false, 6.0)  # 6 is thickness of background grid lines
