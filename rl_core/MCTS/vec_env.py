import numpy as np
import time, os
from rl_core.env import Result
from rich import print

class VecTronEnv:
    """
    Vectorized Tron environment for parallel rollouts.

    State per env:
        walls  (B, size, size)
        pos1   (B, 2)   adversary bike
        pos2   (B, 2)   controlled bike
        h1     (B,)     adversary heading
        h2     (B,)     agent heading
    """

    action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int8)

    reward_dict = {
        Result.DRAW: 0,
        Result.BIKE2_CRASH: -1,
        Result.BIKE1_CRASH: 1,
        Result.PLAYING: 0,
    }

    def __init__(self, num_envs, size, render=False):
        self.num_envs = num_envs
        self.size = size
        if render:
            os.system('cls')
        self.render = render

        self.n_actions = 3  # left, forward, right
        self.obs_shape = (3, size, size)

        self.walls = np.zeros((num_envs, size, size), dtype=np.int8)
        self.pos1 = np.zeros((num_envs, 2), dtype=np.int8)
        self.pos2 = np.zeros((num_envs, 2), dtype=np.int8)

        self.h1 = np.zeros(num_envs, dtype=np.int8)
        self.h2 = np.zeros(num_envs, dtype=np.int8)

        self.env_ids = np.arange(num_envs)

    def sample_actions(self):
        return np.random.randint(self.n_actions, size=self.num_envs, dtype=np.int8)

    def reset(self, mask=None):
        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)

        self.walls[mask] = 0
        self.pos1[mask] = np.array([self.size // 6, self.size // 2])
        self.pos2[mask] = np.array([5 * self.size // 6, self.size // 2])

        self.h1[mask] = 1
        self.h2[mask] = 3

        return VecTronEnv.encode(self.state), {"state": self.state}
    
    def step(self, actions):

        actions = np.asarray(actions, dtype=np.int8)
        B = self.num_envs

        # ----- agent heading update -----
        self.h2 = (self.h2 + (actions - 1)) % 4
        self.h1 = self._adv_policy()
        
        # ----- update walls -----
        self.walls[self.env_ids, self.pos1[:, 1], self.pos1[:, 0]] = 1
        self.walls[self.env_ids, self.pos2[:, 1], self.pos2[:, 0]] = 1

        # ----- movement -----
        move1 = self.action_mapping[self.h1]
        move2 = self.action_mapping[self.h2]

        new_pos1 = self.pos1 + move1
        new_pos2 = self.pos2 + move2

        new_pos1[:, 0] = np.clip(new_pos1[:, 0], 0, self.size - 1)
        new_pos1[:, 1] = np.clip(new_pos1[:, 1], 0, self.size - 1)

        new_pos2[:, 0] = np.clip(new_pos2[:, 0], 0, self.size - 1)
        new_pos2[:, 1] = np.clip(new_pos2[:, 1], 0, self.size - 1)

        # ----- collision detection -----
        crash1 = self._is_hit(new_pos1)
        crash2 = self._is_hit(new_pos2)

        head_on = np.all(new_pos1 == new_pos2, axis=1)

        crash1 = crash1 | head_on
        crash2 = crash2 | head_on

        result = np.full(B, Result.PLAYING, dtype=np.int8)

        result[crash1 & crash2] = Result.DRAW
        result[crash1 & ~crash2] = Result.BIKE1_CRASH
        result[crash2 & ~crash1] = Result.BIKE2_CRASH


        # ----- update positions -----
        self.pos1 = new_pos1
        self.pos2 = new_pos2

        # ----- rewards -----
        reward = np.zeros(B, dtype=np.float32)

        reward[result == Result.BIKE1_CRASH] = 1
        reward[result == Result.BIKE2_CRASH] = -1
        reward[result == Result.DRAW] = 0

        done = result != Result.PLAYING
        obs = VecTronEnv.encode(self.state)
        if self.render:
            self.view()  # render
        self.reset(mask=done)  # Auto reset done envs

        
        infos = {"result": result, "state": self.state}
        return obs, reward, done, None, infos

    def _is_hit(self, pos):
        x = pos[:, 0]
        y = pos[:, 1]

        hit_wall = self.walls[self.env_ids, y, x] == 1
        return hit_wall

    def _adv_policy(self):
        B = self.num_envs
        actions = np.zeros(B, dtype=np.int8)

        for i in range(B):

            walls = self.walls[i]
            bike1 = self.pos1[i]
            head1 = self.h1[i]

            candidates = [0, 1, 2, 3]
            candidates.remove((head1 + 2) % 4)

            chosen = 1

            for a in candidates:
                new_pos = bike1 + self.action_mapping[a]
                x, y = new_pos

                if x < 0 or x >= self.size or y < 0 or y >= self.size:
                    continue

                if walls[y, x] == 0:
                    chosen = a
                    break

            actions[i] = chosen

        return actions

    def set_state(self, state, mask=None):
        walls, p1, p2, h1, h2 = state
        assert walls.shape == (self.size, self.size), f"Expected shape {(self.size, self.size)}, got {walls.shape}"
        assert p1.shape == (2,), f"Expected shape {(2,)}, got {p1.shape}"
        assert p2.shape == (2,), f"Expected shape {(2,)}, got {p2.shape}"
        assert isinstance(h1, int) and isinstance(h2, int), f"Expected integer headings, got {type(h1)} and {type(h2)}" 

        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)
        count = mask.sum()
        if count == 0:
            return  # No envs to set

        self.walls[mask] = np.repeat(walls[None], count, axis=0).copy()
        self.pos1[mask] = np.repeat(p1[None], count, axis=0).copy()
        self.pos2[mask] = np.repeat(p2[None], count, axis=0).copy()

        self.h1[mask] = h1
        self.h2[mask] = h2

    def set_states(self, states):
        walls, p1, p2, h1, h2 = states
        assert walls.shape == (self.num_envs, self.size, self.size), f"Expected shape {(self.num_envs, self.size, self.size)}, got {walls.shape}"
        assert p1.shape == (self.num_envs, 2), f"Expected shape {(self.num_envs, 2)}, got {p1.shape}"
        assert p2.shape == (self.num_envs, 2), f"Expected shape {(self.num_envs, 2)}, got {p2.shape}"
        assert h1.shape == (self.num_envs,), f"Expected shape {(self.num_envs,)}, got {h1.shape}"
        assert h2.shape == (self.num_envs,), f"Expected shape {(self.num_envs,)}, got {h2.shape}"

        self.walls = walls
        self.pos1 = p1
        self.pos2 = p2
        self.h1 = h1
        self.h2 = h2

    @staticmethod
    def encode(state):
        walls, p1, p2, _, h2 = state

        B, size, _ = walls.shape

        obs = np.zeros((B, 3, size, size), dtype=np.float32)

        obs[:, 0] = walls

        env_ids = np.arange(B)

        obs[env_ids, 1, p1[:, 1], p1[:, 0]] = 1
        obs[env_ids, 2, p2[:, 1], p2[:, 0]] = 1

        for i in range(B):
            obs[i] = np.rot90(obs[i], k=h2[i], axes=(1, 2))

        return obs

    @property
    def state(self):
        return (
            self.walls.copy(),
            self.pos1.copy(),
            self.pos2.copy(),
            self.h1.copy(),
            self.h2.copy(),
        )
    
    def view(self):
        os.system('cls')
        # print("\033[H", end="")  # move cursor to top (terminal animation)
        for i in range(self.num_envs):

            board = np.full((self.size, self.size), ".", dtype=str)

            # walls
            board[self.walls[i] == 1] = "#"

            # bikes
            x1, y1 = self.pos1[i]
            x2, y2 = self.pos2[i]

            board[y1, x1] = "A"
            board[y2, x2] = "B"

            print(f"ENV {i}")
            for row in board:
                print(" ".join(row))

            print("=" * (self.size * 2))

        # time.sleep(0.4)

if __name__ == "__main__":
    from rl_core.env.env import TronEnv
    #########################
    ##### Does it tick? #####
    #########################
    SIZE=6
    NUM_ENVS = 7
    envs = VecTronEnv(NUM_ENVS, SIZE)
    envs.reset()
    for steps in range(100):
        actions = envs.sample_actions()
        obs, reward, done, _, _ = envs.step(actions)

    print("[green]Pass[/green]")

    #############################
    ##### Verify set_states #####
    #############################
    SIZE=6
    NUM_ENVS = 64
    envs = VecTronEnv(NUM_ENVS, SIZE)
    envs.reset()
    actions = [0,1,2,3,1,0]
    states = []
    obs_list = []

    for a in actions:
        states.append(envs.state)
        obs, r, done, _, _ = envs.step(a)
        obs_list.append(obs)

    # replay
    envs.set_states(states[2])

    for i,a in enumerate(actions[2:]):
        obs2, r2, done2, _, _ = envs.step(a)

        assert (obs2 == obs_list[2+i]).all()
    print("[green]Pass[/green]")
    
    ######################################################
    #### Vectorized vs Single Environment Consistency ####
    ######################################################
    SIZE = 7
    N = 64
    steps = 20
    np.random.seed(23)

    render=False
    vec = VecTronEnv(N, SIZE, render)
    vec.reset()
    single_envs = [TronEnv(SIZE, None, render) for _ in range(N)]

    for e in single_envs:
        e.reset()
    
    for _ in range(steps):
        actions = vec.sample_actions()
        obs_v, r_v, done_v, _, v_infos = vec.step(actions)

        for i,e in enumerate(single_envs):
            obs_s, r_s, done_s, _, info = e.step(actions[i])
            assert (obs_v[i] == obs_s).all()
            assert done_v[i] == done_s
            assert r_v[i] == r_s, f"Reward mismatch at step {_} for env {i}: vectorized reward {r_v[i]} vs single env reward {r_s}"
            if done_s:
                e.reset()
    print("[green]Pass[/green]")

    ########################
    ### Reset with mask ####
    ########################
    # Reset only envs 0, 2, 4 and verify their walls sum to num_envs
    SIZE=7
    NUM_ENVS = 7
    render=False
    envs = VecTronEnv(NUM_ENVS, SIZE, render)
    obs, _ = envs.reset()

    mask = np.array([True, False, True, False, True, False, False])
    steps = 2  # Can't kill self in 2 steps
    for _ in range(steps):
        actions = envs.sample_actions()
        obs, reward, done, _, _ = envs.step(actions)

    envs.reset(mask=mask)
    assert envs.walls[mask].sum() == 0, f"Expected reset envs to have 3 wall cells, got {envs.walls[mask].sum(axis=(1,2))}"
    assert envs.walls[~mask].sum() > 0, f"Expected non-reset envs to have wall cells, got {envs.walls[~mask].sum(axis=(1,2))}"
    print("[green]Pass[/green]")
