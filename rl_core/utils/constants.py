from enum import Enum

class Color(tuple, Enum):
    BLUE = (34, 49, 63)
    BLUE_ALT = (41, 64, 82)
    GREEN = (20, 180, 20)
    GREEN_ALT = (20, 220, 20)
    RED = (180, 20, 20)
    RED_ALT = (220, 20, 20)

    GREEN_TRANSPARENT = (20, 180, 20, 100)
    RED_TRANSPARENT = (180, 20, 20, 100)
    YELLOW_TRANSPARENT = (255, 255, 0, 100)