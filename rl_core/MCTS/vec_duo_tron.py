# import colorama
# colorama.just_fix_windows_console()
import numpy as np
import time, os, sys
from rl_core.env import Result, GameState
from rich import print
from dataclasses import dataclass

@dataclass(frozen=True)
class VecGameState:
    walls: np.ndarray          # (num_envs, size, size)
    pos1: np.ndarray           # (num_envs, 2)
    pos2: np.ndarray           # (num_envs, 2)
    heading1: np.ndarray  # (num_envs,)
    heading2: np.ndarray  # (num_envs,)

    def __iter__(self):
        return iter((self.walls, self.pos1, self.pos2, self.heading1, self.heading2))

class VecTronDuoEnv:
    """
    Vectorized Duo Tron environment for parallel rollouts.

    State per env:
        walls  (B, size, size)
        pos1   (B, 2)   human bike
        pos2   (B, 2)   controlled bike
        h1     (B,)     human heading
        h2     (B,)     agent heading
    """

    action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int8)

    reward_dict = {  # reward from agent1 perspective
        Result.DRAW: 0,
        Result.BIKE2_CRASH: 1,  # agent1 win
        Result.BIKE1_CRASH: -1,  # agent2 win
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
        self.pos1 = np.zeros((num_envs, 2), dtype=np.int8)  # (x, y) positions
        self.pos2 = np.zeros((num_envs, 2), dtype=np.int8)

        self.h1 = np.zeros(num_envs, dtype=np.int8)
        self.h2 = np.zeros(num_envs, dtype=np.int8)

        self.env_ids = np.arange(num_envs)

    def sample_actions(self):
        return np.random.randint(self.n_actions, size=(self.num_envs, 2), dtype=np.int8)

    def reset(self, mask=None):
        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)

        self.walls[mask] = 0
        self.pos1[mask] = np.array([self.size // 6, self.size // 2])
        self.pos2[mask] = np.array([5 * self.size // 6, self.size // 2])

        self.h1[mask] = 1
        self.h2[mask] = 3

        return VecTronDuoEnv.encode(self.state), {"state": self.state}
    
    def step(self, joint_actions: np.ndarray):
        assert isinstance(joint_actions, np.ndarray), f"Expected joint_actions to be a numpy array, got {type(joint_actions)}"
        assert joint_actions.shape == (self.num_envs, 2), f"Expected joint_actions shape to be {(self.num_envs, 2)}, got {joint_actions.shape}"
        
        B = self.num_envs

        # ----- agent heading update -----
        self.h1 = (self.h1 + (joint_actions[:, 0] - 1)) % 4
        self.h2 = (self.h2 + (joint_actions[:, 1] - 1)) % 4
        
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
        # reward = np.zeros(B, dtype=np.float32)
        reward = np.full(B, self.reward_dict[Result.PLAYING], dtype=np.float32)

        reward[result == Result.BIKE1_CRASH] = self.reward_dict[Result.BIKE1_CRASH]
        reward[result == Result.BIKE2_CRASH] = self.reward_dict[Result.BIKE2_CRASH]
        reward[result == Result.DRAW] = self.reward_dict[Result.DRAW]
        

        dones = result != Result.PLAYING
        if self.render:
            self.view()  # render
        
        state = self.state
        obs = VecTronDuoEnv.encode(state)
        self.reset(mask=dones)  # Auto reset dones envs
        
        infos = {"result": result, "state": state}
        return obs, reward, dones, None, infos

    def _is_hit(self, pos):
        x = pos[:, 0]
        y = pos[:, 1]

        hit_wall = self.walls[self.env_ids, y, x] == 1
        return hit_wall

    def set_state(self, state: GameState, mask=None):
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

    def set_states(self, states: VecGameState):
        walls, p1, p2, h1, h2 = states
        assert walls.shape == (self.num_envs, self.size, self.size), f"Expected shape {(self.num_envs, self.size, self.size)}, got {walls.shape}"
        assert p1.shape == (self.num_envs, 2), f"Expected shape {(self.num_envs, 2)}, got {p1.shape}"
        assert p2.shape == (self.num_envs, 2), f"Expected shape {(self.num_envs, 2)}, got {p2.shape}"
        assert h1.shape == (self.num_envs,), f"Expected shape {(self.num_envs,)}, got {h1.shape}"
        assert h2.shape == (self.num_envs,), f"Expected shape {(self.num_envs,)}, got {h2.shape}"

        self.walls = walls.copy()
        self.pos1 = p1.copy()
        self.pos2 = p2.copy()
        self.h1 = h1
        self.h2 = h2

    @staticmethod
    def encode(state: VecGameState):
        assert isinstance(state, VecGameState), f"Expected VecGameState, got {type(state)}"
        walls, p1, p2, h1, h2 = state
        B, size, _ = walls.shape

        obs1 = np.zeros((B, 3, size, size), dtype=np.float32)
        obs2 = np.zeros((B, 3, size, size), dtype=np.float32)

        obs1[:, 0] = walls
        obs2[:, 0] = walls

        env_ids = np.arange(B)

        obs1[env_ids, 1, p1[:, 1], p1[:, 0]] = 1  
        obs1[env_ids, 2, p2[:, 1], p2[:, 0]] = 1
        obs2[env_ids, 1, p2[:, 1], p2[:, 0]] = 1
        obs2[env_ids, 2, p1[:, 1], p1[:, 0]] = 1

        for i in range(B):
            obs1[i] = np.rot90(obs1[i], k=h1[i], axes=(1, 2))
            obs2[i] = np.rot90(obs2[i], k=h2[i], axes=(1, 2))

        obs = np.stack([obs1, obs2], axis=1)  # shape (B, 2, 3, size, size)
        return obs

    @property
    def state(self) -> VecGameState:
        return VecGameState(
            self.walls.copy(),
            self.pos1.copy(),
            self.pos2.copy(),
            self.h1.copy(),
            self.h2.copy(),
        )
    
    def view(self, flush=True):
        # os.system('cls')
        # print("\033[H", end="")  # move cursor to top (terminal animation)
        # print("\033[H\033[J", end="")
        lines = []
        for i in range(self.num_envs):

            board = np.full((self.size, self.size), ".", dtype=str)

            # walls
            board[self.walls[i] == 1] = "#"

            # bikes
            x1, y1 = self.pos1[i]
            x2, y2 = self.pos2[i]

            board[y1, x1] = "A"
            board[y2, x2] = "B"

            # print(f"ENV {i}")
            lines.append(f"ENV {i}")
            for row in board:
                # print(" ".join(row))
                lines.append(" ".join(row))


            lines.append("=" * (self.size * 2))
            # print("=" * (self.size * 2))
        frame = "\n".join(lines)

        if flush:
            sys.stdout.write("\033[H")
            sys.stdout.write(frame)
            sys.stdout.flush()
            time.sleep(0.4)
        else:
            print(frame)



if __name__ == "__main__":
    from rl_core.env.env import TronDuoEnv
    #########################
    ##### Does it tick? #####
    #########################
    print("Testing basic functionality... ", end="")
    SIZE=6
    NUM_ENVS = 4
    envs = VecTronDuoEnv(NUM_ENVS, SIZE, render=False)
    envs.reset()
    for steps in range(100):
        actions = envs.sample_actions()
        obs, reward, done, _, _ = envs.step(actions)

    print("[green]Pass[/green]")
    
    #############################
    ##### Verify set_states #####
    #############################
    print("Testing set_states consistency... ", end="")
    SIZE=6
    NUM_ENVS = 64
    envs = VecTronDuoEnv(NUM_ENVS, SIZE)
    envs.reset()
    actions = np.random.randint(3, size=(6, NUM_ENVS, 2), dtype=np.int8)
    states = []
    obs_list = []

    for a in actions:
        states.append(envs.state)
        obs, r, done, _, _ = envs.step(a)
        obs_list.append(obs)

    # replay
    envs.set_states(states[2])

    for i, a in enumerate(actions[2:]):
        obs2, r2, done2, _, _ = envs.step(a)

        assert (obs2 == obs_list[2+i]).all()
    print("[green]Pass[/green]")

    ######################################################
    #### Vectorized vs Single Environment Consistency ####
    ######################################################
    print("Testing vectorized vs single environment consistency... ", end="")
    SIZE = 5
    N = 64
    steps = 20
    np.random.seed(23)

    render=False
    vec = VecTronDuoEnv(N, SIZE, render)
    vec.reset()
    single_envs = [TronDuoEnv(SIZE) for _ in range(N)]

    for e in single_envs:
        e.reset()
    
    for _ in range(steps):
        actions = vec.sample_actions()
        obs_v, r_v, done_v, _, v_infos = vec.step(actions)

        for i,e in enumerate(single_envs):
            obs_s, r_s, done_s, _, info = e.step(actions[i])
            assert (obs_v[i] == obs_s).all(), f"{obs_v[i]}\n{obs_s}"
            assert done_v[i] == done_s
            assert r_v[i] == -r_s, f"Reward mismatch at step {_} for env {i}: vectorized reward {r_v[i]} vs single env reward {r_s}"
            if done_s:
                e.reset()
    print("[green]Pass[/green]")

    ########################
    ### Reset with mask ####
    ########################
    print("Testing reset with mask... ", end="")
    # Reset only envs 0, 2, 4 and verify their walls sum to num_envs
    SIZE=7
    NUM_ENVS = 7
    render=False
    envs = VecTronDuoEnv(NUM_ENVS, SIZE, render)
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
