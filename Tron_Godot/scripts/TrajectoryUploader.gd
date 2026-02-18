extends HTTPRequest
class_name TrajectoryUploader

var server_url = "http://127.0.0.1:8000/trajectory"
var queue = []
var is_sending = false

func _ready():
    connect("request_completed", self, "_on_request_completed")

func enqueue_trajectory(history, winner):
    queue.append({"history": history, "winner": winner})
    _send_next()

func _send_next():
    if is_sending or queue.size() == 0:
        return

    is_sending = true

    var item = queue.pop_front()

    var actions = []
    for step in item["history"]:
        actions.append([int(step.x), int(step.y)])

    var payload = {
        "actions": actions,
        "winner": item["winner"]
    }

    var body = to_json(payload)

    var headers = ["Content-Type: application/json"]

    var err = request(server_url, headers, true, HTTPClient.METHOD_POST, body)

    if err != OK:
        push_error("Failed to send request: %s" % err)
        is_sending = false

func _on_request_completed(result, response_code, headers, body):
    print("POST completed. Response code:", response_code)
    is_sending = false
    _send_next()
