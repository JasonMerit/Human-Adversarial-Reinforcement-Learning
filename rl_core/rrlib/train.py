import os
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import ray
import gymnasium as gym

from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.dqn.dqn_catalog import DQNCatalog

from .env import TronMultiAgentEnv

OBS = TronMultiAgentEnv.OBS
ACT = TronMultiAgentEnv.ACT

rl_module_spec = RLModuleSpec(
    observation_space=OBS,
    action_space=ACT,
    catalog_class=DQNCatalog,
    model_config=dict(conv_filters=[[32, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1]], conv_activation="relu")
    )
config = (
    DQNConfig()
    .environment(TronMultiAgentEnv)
    .env_runners(num_env_runners=1)
    .framework("torch")
    .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
    .rl_module(rl_module_spec=rl_module_spec)
    .training(
        gamma=0.99,
        lr=1e-4,
        train_batch_size=1024,
        target_network_update_freq=8000,
        double_q=True,
        dueling=True,
        n_step=1,
        num_atoms=1,
        noisy=False,
        replay_buffer_config=dict(
            type="MultiAgentPrioritizedReplayBuffer",
            capacity=200000,
            alpha=0.6,
            beta=0.4,
        ),
    )
    .multi_agent(
        policies=dict(shared_policy=(None, OBS, ACT, {})),
        policy_mapping_fn=lambda *_: "shared_policy",
    )
)

ray.init()

algo = config.build_algo()

for i in range(100000):
    result = algo.train()
    print(i, result["episode_reward_mean"])

algo.save()