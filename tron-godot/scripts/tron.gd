class_name Tron

class Bike:
	var pos: Vector2i
	var dir: Vector2i
	var last_pos: Vector2  # Set by main.gd for interpolation

	func _init(start_pos: Vector2):
		pos = start_pos
		dir = Vector2.ZERO  # For interpolation set by main.gd

	func move(vel: Vector2i, walls: Array) -> bool:
		# Returns true if crash
		var new_pos = pos + vel
		
		if is_hit(walls, new_pos.x, new_pos.y):
			return true
		
		pos = new_pos
		return false


	func is_hit(walls: Array, x: int, y: int) -> bool:
		# Out of bounds
		if y < 0 or y >= walls.size():
			return true
		if x < 0 or x >= walls[0].size():
			return true
		
		# Wall collision
		return walls[y][x] != 0

var width: int
var height: int
var walls: Array
var bike1: Tron.Bike
var bike2: Tron.Bike

func _init(grid_size: Vector2i):
	width = grid_size.x
	height = grid_size.y
	reset()


func reset() -> void:
	# Create 2D grid
	walls = []
	for y in range(height):
		var row = []
		for x in range(width):
			row.append(0)
		walls.append(row)
	
	bike1 = Bike.new(Vector2i(1, height >> 1))
	bike2 = Bike.new(Vector2i(width - 2, height >> 1))



func tick(dir1: Vector2i, dir2: Vector2i) -> int:
	# Place trails
	walls[bike1.pos.y][bike1.pos.x] = 1
	walls[bike2.pos.y][bike2.pos.x] = 2
	
	var bike1_hit = bike1.move(dir1, walls)
	var bike2_hit = bike2.move(dir2, walls)

	# Both collided or head-on
	if (bike1_hit and bike2_hit) or bike1.pos == bike2.pos:
		return 0
	
	if bike1_hit:
		return 2
	
	if bike2_hit:
		return 1
	
	return -1



