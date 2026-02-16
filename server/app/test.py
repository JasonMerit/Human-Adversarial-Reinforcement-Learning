from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_act():
    payload = {
        "grid": [[0]*11 for _ in range(11)],
        "bike1": [1, 5],
        "bike2": [9, 5]
    }

    response = client.post("/act", json=payload)
    print(response.status_code)
    print(response.json())

test_act()