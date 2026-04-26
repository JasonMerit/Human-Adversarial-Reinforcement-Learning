# From https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rainbow_atari.py
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rainbow/#rainbow_ataripy
import random, os, time, shutil

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from rich import print
import yaml

from .argp import read_args
from .agents import RainbowAgent, MCTSAgent
from .agents.utils import TimerRegistry
from .env import TronDuoEnv, TronView, PoLEnv
from rl_core.MCTS.vec_pol import VecPoLEnv

def make_env(idx, args):
    def thunk():
        Env = TronDuoEnv if not args.pol else PoLEnv
        env = Env(args.size)
        if args.render and idx == 0:
            env = TronView(env, fps=100000)
        env.action_space.seed(args.seed + idx)
        return env
    return thunk

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    args = read_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    writer = None
    if args.track:
        i = 0
        if args.hpc:
            if "[" in args.exp_name:  # If using job arrays, remove brackets for folder name
                args.exp_name = args.exp_name.split("[")[0]
            save_folder = f"rl_core/HPC/runs/{args.exp_name}"  # Move to HPC folder
        else:
            save_folder = f"runs/{args.exp_name}"

        while os.path.exists(save_folder + f"_{i}"):
            i += 1
        save_folder += f"_{i}/"
        os.makedirs(save_folder)
        with open(save_folder + "args.yml", "w") as f:
            yaml.dump(vars(args), f)
        
        run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
        writer = SummaryWriter(save_folder)
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

        if args.save:
            print(f"Models will be saved to {save_folder}")
        else:
            print("Models will NOT be saved!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=====| {args.exp_name} on {device} with seed {args.seed}", "[yellow bold](debug mode)[/yellow bold]" if args.debug else "", "|=====")

    # Handle parallel envs
    total_loops = args.total_timesteps // args.num_envs
    learn_start_loop = args.learning_starts // args.num_envs
    target_every = args.target_network_frequency // args.num_envs
    train_count = args.num_envs // args.train_frequency
    log_every = max(1, total_loops // 100)
    save_every = max(1, total_loops // args.total_checkpoints)
    

    # Envs and agents
    if args.vec:
        assert args.pol, "Vectorized environments are only implemented for the PoL environment"
        envs = VecPoLEnv(args.num_envs, args.size, args.render)
        obs_shape = envs.obs_shape
        n_actions = envs.n_actions
    else:
        envs = gym.vector.SyncVectorEnv([make_env(i, args) for i in range(args.num_envs)])
        obs_shape = envs.single_observation_space.shape
        n_actions = envs.single_action_space.n

    obs, infos = envs.reset()
    state = infos["state"]

    # envs = gym.vector.SyncVectorEnv([make_env(i, args) for i in range(args.num_envs)])
    # obs_shape = envs.single_observation_space.shape[-3:]  # Ignore the player channel
    # n_actions = envs.single_action_space.nvec[0] if not args.pol else envs.single_action_space.n
    print(f"Observation shape: {obs_shape}, Action space: {n_actions}")

    # agent1 = RainbowAgent(obs_shape, n_actions, args, device, writer, "A")
    Agent = MCTSAgent if args.mcts else RainbowAgent
    agent1 = Agent(obs_shape, n_actions, state, envs.encode, args, device, writer, "A")

    # PoL Specific
    # env_eval = PoLEnv(args.size, True) if args.pol else None
    env_eval = PoLEnv(args.size) if args.pol else None
    eval_every = 10  # Evaluate every 10 learning steps
    shortest_path = float('inf')
    win_combo = 0
    from rl_core.eval.pol_eval import eval

    # Logging
    # TimerRegistry.disable()
    start_time = time.time()
    results = [0, 0, 0]
    total_episodes = 0
    # total_episode_lengths = 0
    episode_lengths = np.zeros(args.num_envs, dtype=int)

    from collections import deque
    ep_lens = deque(maxlen=100)

    
    for global_step in range(1, total_loops + 1):
        # agent1.q_network.reset_noise()
        # agent2.q_network.reset_noise()
        # obs1, obs2 = obs[:, 0], obs[:, 1]
        a1 = agent1.select_action(obs)
        # a1 = agent1.act(obs1)
        # a2 = np.random.randint(0, n_actions, size=args.num_envs)  # Random actions for agent2
        # a2 = agent2.act(obs2)

        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * total_loops, global_step)
        explore_mask = np.random.rand(args.num_envs) < epsilon
        a1[explore_mask] = np.random.randint(0, n_actions, size=explore_mask.sum())
        # explore_mask = np.random.rand(args.num_envs) < epsilon
        # a2[explore_mask] = np.random.randint(0, n_actions, size=explore_mask.sum())

        # actions = np.stack([a1, a2], axis=1) 
        actions = a1

        next_obs, rewards, dones, _, infos = envs.step(actions)
        next_state = infos["state"]
        agent1.rb.add(state, a1, rewards, next_state, dones)
        # agent1.rb.add(obs, a1, rewards, next_obs, dones, infos)
        # agent1.rb.add(obs1, a1, rewards, next_obs1, dones)
        # agent2.rb.add(obs2, a2, -rewards, next_obs2, dones)

        # obs = next_obs
        obs, state = next_obs, next_state
        episode_lengths += 1

        # Training
        if global_step > learn_start_loop:
            # for _ in range(train_count):
            agent1.learn()
            # if agent1.learning_steps > 10:
            #     break
                # agent2.learn()

            # update target network
            if global_step % target_every == 0:
                agent1.update_target()
                # agent2.update_target()
            
            if global_step % eval_every == 0:
                eval_result = eval(agent1.q_network, env_eval, device)

                shortest_path = min(shortest_path, eval_result)
                if eval_result == env_eval.size * 2 - 2:  # Shortest path in an empty grid is size*2 - 2
                    win_combo += 1
                    if win_combo == 10:  # If the agent has solved the environment 10 times in a row
                        print("[bold green]Agent has consistently solved the environment!")
                        break
                else:
                    win_combo = 0
        
        
        # Logging
        for i in np.where(dones)[0]:  # Update results for any env that is done
            # results[infos["result"][i]] += 1
            # total_episode_lengths += episode_lengths[i]
            ep_lens.append(episode_lengths[i])
            total_episodes += 1
            episode_lengths[i] = 0
        #     if writer:
        #         writer.add_scalar("charts/draw_percentage", results[0] / total_episodes, total_episodes)
        #         writer.add_scalar("charts/agent1_win_percentage", results[1] / total_episodes, total_episodes)
        #         writer.add_scalar("charts/agent2_win_percentage", results[2] / total_episodes, total_episodes)
        #         writer.add_scalar("charts/avg_episode_length", total_episode_lengths / total_episodes, total_episodes)

        if global_step % log_every == 0:
            sps = int(global_step * args.num_envs / (time.time() - start_time))
            elapsed = time.time() - start_time
            progress = global_step / total_loops
            eta = elapsed * (1/progress - 1)
            # print(f"{progress*100:.1f}% - {epsilon=:.3f}")
            # epi_len = total_episode_lengths / total_episodes if total_episodes > 0 else 0
            avg = sum(ep_lens) / 100
            print(f"{progress*100:.1f}% - SPS: {sps} - epi_len: {avg:.2f} - eval_len {shortest_path} (x{win_combo}) - {eta/60:.1f} minutes left...")
            # print(f"{progress*100:.1f}% - SPS: {sps} - Results: {results} - epi_len: {epi_len:.2f} - {eta/60:.1f} minutes left...")
        
        # env_step = global_step * args.num_envs
        # if args.save and global_step % save_every == 0:
        #     agent1.save(save_folder + f"A_{env_step}.pth")
        #     agent2.save(save_folder + f"B_{env_step}.pth")

    envs.close()
    TimerRegistry.report()

    if args.track:
        if args.save:
            agent1.save(save_folder + f"A.pth")
            # agent2.save(save_folder + f"B.pth")

        with open(save_folder + "results.yml", "w") as f:
            yaml.dump({
                "results": results, 
                "steps_taken": global_step * args.num_envs,
                "global_steps": global_step,
                "training_time_mins": (time.time() - start_time) / 60,
                }, f)
        writer.close()
        TimerRegistry.export(save_folder + "timers.json")

        if args.hpc:  # Duplicate logs
            shutil.copy(f"rl_core/HPC/Out_{args.job_index}.out", save_folder + "Out.out")
            shutil.copy(f"rl_core/HPC/Err_{args.job_index}.err", save_folder + "Err.err")

    # if isinstance(agent1, RainbowAgent):        
    #     weights = agent1.q_network.adv_head[-1].weight.data.cpu().numpy().mean(axis=1)
    #     expected = np.array( [ 0.00032227,  0.03281876,  0.03462983, -0.00111074])
    #     # print("Agent1 last layer weights:", agent1.q_network.adv_head[-1].weight.data.cpu().numpy().mean(axis=1))
    #     assert np.allclose(weights, expected, atol=1e-2), f"Unexpected weights {weights}, expected {expected}"
