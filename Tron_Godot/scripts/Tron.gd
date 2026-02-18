class Bike:
    var pos
    var dir
    var last_pos  # Used for interpolation (Vector2 in main)

    func _init(start_pos):
        pos = start_pos
        dir = Vector2.ZERO

    func move(vel, walls):
        # Returns true if crash
        var new_pos = pos + vel

        if is_hit(walls, int(new_pos.x), int(new_pos.y)):
            return true

        pos = new_pos
        return false

    func is_hit(walls, x, y):
        # Out of bounds
        if y < 0 or y >= walls.size():
            return true
        if x < 0 or x >= walls[0].size():
            return true

        # Wall collision
        return walls[y][x] != 0


var width
var height
var walls
var bike1
var bike2

func _init(grid_size):
    width = int(grid_size.x)
    height = int(grid_size.y)
    reset()

func reset():
    walls = []

    for y in range(height):
        var row = []
        for x in range(width):
            row.append(0)
        walls.append(row)

    bike1 = Bike.new(Vector2(1, height / 2))
    bike2 = Bike.new(Vector2(width - 2, height / 2))

func tick(dir1, dir2):
    # Place trails
    walls[int(bike1.pos.y)][int(bike1.pos.x)] = 1
    walls[int(bike2.pos.y)][int(bike2.pos.x)] = 2

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
