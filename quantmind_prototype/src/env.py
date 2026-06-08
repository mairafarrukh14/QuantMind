"""QuantMind portfolio-management environment (Gymnasium).

This frames active portfolio management as the "decision making under
uncertainty" problem the template suggests, in the style of Jiang et al. (2017):
the agent observes engineered technical features for every asset plus its own
current allocation, and outputs a *target portfolio weight vector* over the
assets and cash. The reward is the realised log return of the portfolio net of
transaction costs, minus a volatility penalty for risk-adjustment.

State  (Box):  flattened [features (n_assets x n_features)] + [current weights (n_assets + 1)]
Action (Box):  raw logits (n_assets + 1) -> softmax -> next-day target weights
Reward:        log(value_t / value_{t-1}) - cost - RISK_PENALTY * rolling_vol
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config


def apply_position_cap(weights: np.ndarray, cap: float, n_assets: int) -> np.ndarray:
    """Enforce a per-asset concentration limit, redistributing excess weight.

    weights: simplex over [assets..., cash]. Any asset above `cap` is clipped and
    its excess is redistributed proportionally across the remaining uncapped
    assets (or to cash if none remain). A standard risk constraint that turns the
    agent into a genuinely diversified active manager rather than a single bet.
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    assets, cash = w[:n_assets], w[n_assets]
    for _ in range(n_assets):
        over = assets > cap
        if not over.any():
            break
        excess = float((assets[over] - cap).sum())
        assets[over] = cap
        under = assets < cap
        room = assets[under].sum()
        if room > 1e-9:
            assets[under] += excess * assets[under] / room
        else:
            cash += excess
            break
    out = np.concatenate([assets, [cash]])
    return (out / (out.sum() + 1e-12)).astype(np.float32)


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, features: np.ndarray, log_rets, cost: float = config.TRANSACTION_COST,
                 risk_penalty: float = config.RISK_PENALTY,
                 turnover_penalty: float = config.TURNOVER_PENALTY, window_for_vol: int = 20):
        super().__init__()
        # features: (T, n_assets, n_features); log_rets: DataFrame (T, n_assets)
        self.features = features.astype(np.float32)
        self.simple_rets = (np.exp(log_rets.values) - 1.0).astype(np.float32)  # arithmetic
        self.dates = log_rets.index
        self.T, self.n_assets, self.n_features = features.shape
        self.cost = cost
        self.risk_penalty = risk_penalty
        self.turnover_penalty = turnover_penalty
        self.window_for_vol = window_for_vol

        n_weights = self.n_assets + 1  # +1 for cash
        obs_dim = self.n_assets * self.n_features + n_weights
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        # Unbounded logits; softmax inside step() turns them into a simplex.
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(n_weights,), dtype=np.float32)

        self._t = 0
        self.weights = None
        self._ret_history: list[float] = []

    # ------------------------------------------------------------------ #
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        e = np.exp(z)
        return e / (np.sum(e) + 1e-12)

    def _get_obs(self) -> np.ndarray:
        feat = self.features[self._t].reshape(-1)
        return np.concatenate([feat, self.weights]).astype(np.float32)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        # Start fully in cash (last slot is cash).
        self.weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.weights[-1] = 1.0
        self._ret_history = []
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        t = self._t
        target = self._softmax(np.asarray(action, dtype=np.float64))
        target = apply_position_cap(target, config.MAX_WEIGHT, self.n_assets)

        # Transaction cost proportional to turnover (how much we rebalance).
        turnover = np.sum(np.abs(target - self.weights))
        cost = self.cost * turnover

        # CRITICAL (no lookahead): a decision taken at the close of day t earns
        # the return from t -> t+1. The observation features[t] are all known at
        # the close of day t; the reward uses simple_rets[t+1], which is not.
        asset_rets = self.simple_rets[t + 1]                         # (n_assets,)
        port_ret = float(np.dot(target[:-1], asset_rets)) + target[-1] * config.RISK_FREE_DAILY
        net_ret = port_ret - cost

        # Weights drift with the realised returns over the day.
        grown = np.concatenate([target[:-1] * (1.0 + asset_rets), [target[-1]]])
        self.weights = (grown / (np.sum(grown) + 1e-12)).astype(np.float32)

        # Reward = log growth minus a risk penalty (rolling realised vol).
        log_growth = np.log(max(1.0 + net_ret, 1e-6))
        self._ret_history.append(net_ret)
        if len(self._ret_history) >= self.window_for_vol:
            vol = float(np.std(self._ret_history[-self.window_for_vol:]))
        else:
            vol = 0.0
        reward = log_growth - self.risk_penalty * vol - self.turnover_penalty * turnover

        self._t += 1
        terminated = self._t >= self.T - 1
        info = {"port_ret": net_ret, "turnover": float(turnover),
                "target_weights": target, "date": self.dates[t + 1]}
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(reward), terminated, False, info
