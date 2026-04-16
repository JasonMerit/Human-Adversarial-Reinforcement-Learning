from colorama import init
init()

from gymnasium.envs.registration import register, registry


def safe_register(id, entry_point):
    if id not in registry:
        register(id=id, entry_point=entry_point)
safe_register(id="Tron-v0", entry_point="tron_env.env:TronEnv")
safe_register(id="TronDuo-v0", entry_point="tron_env.env:TronDuoEnv")

from .env import TronEnv, TronDuoEnv, Tron2ChannelEnv, PoLEnv
from .utils import StateViewer
from .wrappers import TronView