import time
import threading 
from pynput.mouse import Controller, Button
from pynput.keyboard import Listener, KeyCode

TOGGLE_KEY = KeyCode(char="p")
STOP_KEY = KeyCode(char="q")
UP_KEY = KeyCode(char="2")  # 1
DOWN_KEY = KeyCode(char="1")  # 1

clicking = False
running = True
mouse = Controller()

click_rate = 12  # per second

def clicker():
    global click_rate
    while running:
        if clicking:
            mouse.click(Button.left, 1)
            time.sleep(1/click_rate)  # Click every 1/click_rate seconds
        else:
            time.sleep(0.01)

def toggle_event(key):
    global clicking, running, click_rate
    if key == TOGGLE_KEY:
        clicking = not clicking
    elif key == STOP_KEY:
        running = False
        return False
    elif key == UP_KEY:
        click_rate += 1
        print(f"Click rate increased to {click_rate} clicks/sec")
    elif key == DOWN_KEY:
        click_rate = max(1, click_rate - 1)
        print(f"Click rate decreased to {click_rate} clicks/sec")

print("Press 'p' to toggle clicking, 'q' to quit.")
click_thread = threading.Thread(target=clicker, daemon=True)
click_thread.start()

with Listener(on_press=toggle_event) as listener:
    listener.join()

click_thread.join()