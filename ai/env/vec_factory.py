"""Builds SB3 vectorized envs out of CurveEnv. Headless by construction (CurveEnv
never opens a window), so this is what actually delivers "maximum simulation
speed, no rendering, no graphics window" from the brief: N env copies stepping
in parallel worker processes, each just producing numpy arrays.

Note for curriculum.py/league.py: when `use_subprocess=True`, `opponent_factory`
is sent to each worker process via pickling (both at startup and on any later
`vec_env.env_method("set_opponent_factory", fn)` call) - keep factories as small,
picklable callables (e.g. dataclasses describing *what* to build, reconstructing
Controllers inside `__call__`) rather than closures over live objects.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from ai.env.curve_env import CurveEnv, CurveEnvConfig, OpponentFactory


def make_env_fn(opponent_factory: OpponentFactory, config: CurveEnvConfig, seed: int, monitor_path: str | None = None):
    def _init() -> Monitor:
        env = CurveEnv(opponent_factory, config=config, seed=seed)
        return Monitor(env, filename=monitor_path)

    return _init


def build_vec_env(
    n_envs: int,
    opponent_factory: OpponentFactory,
    config: CurveEnvConfig | None = None,
    base_seed: int = 0,
    use_subprocess: bool = True,
    monitor_dir: str | None = None,
) -> VecEnv:
    config = config or CurveEnvConfig()
    if monitor_dir is not None:
        Path(monitor_dir).mkdir(parents=True, exist_ok=True)
    env_fns = [
        make_env_fn(
            opponent_factory,
            config,
            base_seed + i,
            monitor_path=str(Path(monitor_dir) / f"env_{i}") if monitor_dir else None,
        )
        for i in range(n_envs)
    ]
    if use_subprocess and n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)
