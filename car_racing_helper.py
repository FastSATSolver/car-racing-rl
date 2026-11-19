import os

import gymnasium as gym

from gymnasium.wrappers import GrayscaleObservation, ResizeObservation

from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack


def gray_resize(env, grayscale, resize):
    env = GrayscaleObservation(env, keep_dim=True) if grayscale else env
    env = ResizeObservation(env, resize)
    return env


def make_env(n_envs=1, render_mode=None, n_stack=4, grayscale=True, resize=(84, 84)):
    env = make_vec_env(
        "CarRacing-v3",
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=gray_resize,
        wrapper_kwargs=dict(grayscale=grayscale, resize=resize),
        env_kwargs=dict(lap_complete_percent=0.95,
                        domain_randomize=False,
                        continuous=False,
                        render_mode=render_mode
                        ),
    )
    env = VecFrameStack(env, n_stack=n_stack)
    print(env.observation_space.shape)
    return env


def load_model(env, force_reset=False, device="cuda"):
    if force_reset or not os.path.isfile("carracing.zip"):
        return PPO(
            "CnnPolicy", env,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            verbose=1,
            tensorboard_log="./ppo_carracing_tensorboard/",
            device=device,
        )
    return PPO.load("carracing.zip", env)
