class_name Bike

var pos: Vector2i

func _init(start_pos: Vector2i):
	pos = start_pos


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
