class_name Tron

var width: int
var height: int
var walls: Array
var bike1: Bike
var bike2: Bike


func _init(w: int, h: int):
	width = w
	height = h
	reset()


func reset() -> void:
	# Create 2D grid
	walls = []
	for y in range(height):
		var row = []
		for x in range(width):
			row.append(0)
		walls.append(row)

	bike1 = Bike.new(Vector2i(1, height / 2 as int))
	bike2 = Bike.new(Vector2i(width - 2, height / 2 as int))



func tick(dir1: Vector2i, dir2: Vector2i) -> int:
	# Place trails
	walls[bike1.pos.y][bike1.pos.x] = 1
	walls[bike2.pos.y][bike2.pos.x] = 2
	
	var bike1_hit = bike1.move(dir1, walls)
	var bike2_hit = bike2.move(dir2, walls)

	# Both collided or head-on
	if (bike1_hit and bike2_hit) or bike1.pos == bike2.pos:
		return 3
	
	if bike1_hit:
		return 1
	
	if bike2_hit:
		return 2
	
	return 0
