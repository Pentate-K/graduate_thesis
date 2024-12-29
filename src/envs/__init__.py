import os
import sys

from gymnasium import register
from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper
# from .smaclite_wrapper import SMACliteWrapper
from .grid_env import MultiAgentGridEnv
import importlib.util
# run.py のパスを指定
module_path = os.path.abspath("../epymarl/src/run.py")

# モジュールをロード
spec = importlib.util.spec_from_file_location("custom_run", module_path)
run_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_module)


if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


def __check_and_prepare_smac_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs


# def smaclite_fn(**kwargs) -> MultiAgentEnv:
#     kwargs = __check_and_prepare_smac_kwargs(kwargs)
#     return SMACliteWrapper(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


REGISTRY = {}
# REGISTRY["smaclite"] = smaclite_fn
REGISTRY["gymma"] = gymma_fn
# REGISTRY["grid_obs"] = SimpleGridEnv_Obs

# register(
#     id="grid_obs-v1",                         # Environment ID.
#     entry_point="envs.grid_obs:SimpleGridEnv_Obs",  # The entry point for the environment class
#     kwargs={
#             "grid_size": 5,
#             "obstacles": [(1, 1), (2, 2), (3, 3)]
#             # Arguments that go to MyEnvironment's __init__ function.
#         },
#     )
grid_map = [[0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,0,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0]]
# if(run_module.episode <= 5000):
#     grid_map = []
# else:
#     grid_map = []
# print(grid_map[3][4])
# print(run_module.episode)
# print(run_module.episode_list)
register(
    id="grid_env",
    entry_point="envs.grid_env:MultiAgentGridEnv",  # The entry point for the environment class
    kwargs={
            "grid_map": grid_map,
            # "grid_width": 6,
            "num_agents": 6
            # Arguments that go to MyEnvironment's __init__ function.
        },
)

register(
    id="new_grid_env",
    entry_point="envs.new_grid_env:MultiAgentGridEnv",  # The entry point for the environment class
    kwargs={
            "grid_map": grid_map,
            # "grid_width": 6,
            "num_agents": 4
            # Arguments that go to MyEnvironment's __init__ function.
        },
)


# registering both smac and smacv2 causes a pysc2 error
# --> dynamically register the needed env
def register_smac():
    from .smac_wrapper import SMACWrapper

    def smac_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(**kwargs)

    REGISTRY["sc2"] = smac_fn


def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper

    def smacv2_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACv2Wrapper(**kwargs)

    REGISTRY["sc2v2"] = smacv2_fn
