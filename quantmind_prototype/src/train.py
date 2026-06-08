"""Train the PPO agent on the QuantMind portfolio environment.

PPO (Schulman et al., 2017) via Stable-Baselines3 is a robust on-policy actor-
critic method well suited to the continuous weight-vector action space here.
"""
from __future__ import annotations

import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from . import config
from .env import PortfolioEnv


def make_env(features, log_rets):
    def _init():
        return Monitor(PortfolioEnv(features, log_rets))
    return _init


def train_agent(train_features, train_rets, total_timesteps=config.TOTAL_TIMESTEPS,
                seed=config.SEED, save_path=None, verbose=1):
    venv = DummyVecEnv([make_env(train_features, train_rets)])

    model = PPO(
        "MlpPolicy", venv, seed=seed, verbose=verbose,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    if save_path is None:
        save_path = os.path.join(config.MODELS_DIR, "ppo_quantmind")
    model.save(save_path)
    return model


def load_agent(path=None):
    if path is None:
        path = os.path.join(config.MODELS_DIR, "ppo_quantmind")
    return PPO.load(path)
