extends HTTPRequest

var server_url = "http://127.0.0.1:8000/trajectory"

var queue = []
var is_sending = false

var MAX_RETRIES = 5
var BASE_DELAY = 1.0
var MAX_DELAY = 30.0

func _ready():
    connect("request_completed", self, "_on_request_completed")

func enqueue_trajectory(history, winner):
    queue.append({"history": history, "winner": winner, "retries": 0})
    _send_next()

func _send_next():
    if is_sending or queue.size() == 0:
        return
    is_sending = true

    var item = queue[0]
    var actions = []
    for step in item["history"]:
        actions.append([int(step.x), int(step.y)])

    var payload = {"actions": actions, "winner": item["winner"]}
    var body = to_json(payload)
    var headers = ["Content-Type: application/json"]

    var err = request(server_url, headers, true, HTTPClient.METHOD_POST, body)
    if err != OK:
        _handle_failure()

func _on_request_completed(result, response_code, headers, body):
    is_sending = false

    if result == OK and response_code >= 200 and response_code < 300:
        queue.pop_front()
        _send_next()
    else:
        _handle_failure()

func _handle_failure():
    if queue.size() == 0:
        return

    var item = queue[0]
    item["retries"] += 1
    if item["retries"] > MAX_RETRIES:
        push_error("Dropping trajectory after max retries")
        queue.pop_front()
        _send_next()
        return

    var delay = min(BASE_DELAY * pow(2, item["retries"]), MAX_DELAY)
    yield(get_tree().create_timer(delay), "timeout")
    _send_next()
