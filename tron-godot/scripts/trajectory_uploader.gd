extends HTTPRequest
class_name TrajectoryUploader

var server_url: String = "http://127.0.0.1:8000/trajectory"
var queue: Array = []
var is_sending := false

func _ready():
    request_completed.connect(self._on_request_completed)

func enqueue_trajectory(history: Array, winner: int) -> void:
    queue.append({"history": history, "winner": winner})
    _send_next()

func _send_next() -> void:
    if is_sending or queue.size() == 0:
        return
    is_sending = true

    var item = queue.pop_front()
    var actions = []
    for step in item.history:
        actions.append([int(step.x), int(step.y)])

    var payload = {"actions": actions, "winner": item.winner}
    var body = JSON.stringify(payload)
    var err = request(server_url, [], HTTPClient.METHOD_POST, body)
    if err != OK:
        push_error("Failed to send request: %s" % err)

func _on_request_completed(result, response_code, headers, body):
    print("POST completed. Response code:", response_code)
    is_sending = false
    _send_next()
