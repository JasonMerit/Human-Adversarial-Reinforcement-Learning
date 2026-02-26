from gymnasium.envs.registration import register

register(
    id="Tron-v0",
    entry_point="tron_env.env:TronEnv",
)