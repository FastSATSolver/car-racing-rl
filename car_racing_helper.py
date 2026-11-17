import os

from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack


def make_env(n_envs=1, render_mode=None):
    env = make_vec_env(
        "CarRacing-v3",
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(lap_complete_percent=0.95,
                        domain_randomize=False,
                        continuous=False,
                        render_mode=render_mode
                        ),
    )
    env = VecFrameStack(env, n_stack=4)
    return env


def load_model(env, force_reset=False):
    if force_reset or not os.path.isfile("carracing.zip"):
        return PPO(
            "CnnPolicy", env,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            verbose=1,
            tensorboard_log="./ppo_carracing_tensorboard/",
            device="cuda",
        )
    return PPO.load("carracing.zip", env)
