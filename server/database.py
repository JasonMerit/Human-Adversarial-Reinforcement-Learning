import sqlite3

DB_PATH = "server/trajectories.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            winner INTEGER,
            length INTEGER
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY,
            game_id INTEGER,
            t INTEGER,
            action_p1 INTEGER,
            action_p2 INTEGER,
            FOREIGN KEY(game_id) REFERENCES games(id)
        )
        """)
        con.commit()

def get_last_32_trajectories():
    with get_connection() as con:
        cur = con.cursor()

        cur.execute("""
            SELECT id, winner
            FROM games
            ORDER BY id DESC
            LIMIT 32
        """)
        games = cur.fetchall()

        result = []

        for game_id, winner in games:
            cur.execute("""
                SELECT action_p1, action_p2
                FROM steps
                WHERE game_id = ?
                ORDER BY t ASC
            """, (game_id,))
            
            steps = cur.fetchall()  # [(a1, a2), ...]

            result.append({
                "game_id": game_id,
                "winner": winner,
                "actions": steps
            })

        return result

import pickle

def trajectories_to_pkl(trajectories, save_path="server/last32.pkl"):
    replay_buffer = []

    for episode in trajectories:
        steps = episode["actions"]

        # enforce integer + tuple consistency
        formatted = [(int(a1), int(a2)) for a1, a2 in steps]

        replay_buffer.append(formatted)

    with open(save_path, "wb") as f:
        pickle.dump(replay_buffer, f)

    return replay_buffer


if __name__ == "__main__":
    # init_db()
    trajectories = get_last_32_trajectories()
    trajectories_to_pkl(trajectories)