"""Wiring tests for the algorithm factory (ai/v1_0/models/algo.py), the per-algorithm
policy kwargs, save/load class dispatch, a tiny learn() smoke per algorithm, and
a miniature end-to-end behavior-cloning run. Everything runs on CPU with tiny
observation sizes - these prove the plumbing, not the learning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai.core.env.curve_env import CurveEnv, CurveEnvConfig, RewardConfig
from ai.core.env.observation import ObsConfig
from ai.core.env.opponents import EpisodeConfig
from ai.core.env.vec_factory import build_vec_env
from ai.v1_0.models.algo import ALGO_CHOICES, build_algo_kwargs, load_trained, resolve_algo, sniff_algo_name
from ai.v1_0.models.policy import build_policy_kwargs


def _tiny_env_config() -> CurveEnvConfig:
    return CurveEnvConfig(
        engine_resolution=256,
        obs=ObsConfig(obs_resolution=48, frame_stack=2, n_rays=4, ray_range_px=32.0, arc_horizon=12),
        enabled_items=set(),
        max_episode_ticks=200,
        reward=RewardConfig(),
    )


def _tiny_vec_env():
    return build_vec_env(
        n_envs=1,
        opponent_factory=lambda: EpisodeConfig(opponents=[], enabled_items=set()),
        config=_tiny_env_config(),
        base_seed=0,
        use_subprocess=False,
    )


def _tiny_model(algo: str, env):
    cfg = {
        "algo": algo,
        "ppo": {"n_steps": 32, "batch_size": 32, "n_epochs": 1, "learning_rate": 3e-4},
        "qrdqn": {"buffer_size": 300, "learning_starts": 16, "batch_size": 16, "train_freq": 4, "target_update_interval": 64},
    }
    spec = resolve_algo(cfg)
    return spec.cls(
        spec.policy_name,
        env,
        policy_kwargs=build_policy_kwargs(cnn_arch="small", algo=spec.name),
        verbose=0,
        seed=0,
        **build_algo_kwargs(cfg, spec),
    )


# ------------------------------------------------------------------ resolution


def test_resolve_algo_names_and_masking():
    assert resolve_algo({}).name == "ppo"
    assert resolve_algo({"algo": "recurrent_ppo"}).policy_name == "MultiInputLstmPolicy"
    assert resolve_algo({"algo": "qrdqn"}).is_off_policy
    # masking auto-upgrades ppo -> maskable_ppo (existing behavior)...
    assert resolve_algo({"algo": "ppo", "use_action_masking": True}).name == "maskable_ppo"
    # ...but is rejected loudly for families without a masking path
    for bad in ("recurrent_ppo", "qrdqn"):
        with pytest.raises(SystemExit):
            resolve_algo({"algo": bad, "use_action_masking": True})
    with pytest.raises(SystemExit):
        resolve_algo({"algo": "does_not_exist"})


def test_policy_kwargs_per_algo():
    assert build_policy_kwargs(algo="ppo")["net_arch"] == dict(pi=[128, 128], vf=[128, 128])
    assert build_policy_kwargs(algo="qrdqn")["net_arch"] == [128, 128]  # DQN family: list, not pi/vf dict
    with pytest.raises(ValueError):
        build_policy_kwargs(algo="qrdqn", net_arch=dict(pi=[64], vf=[64]))
    rec = build_policy_kwargs(algo="recurrent_ppo", lstm_kwargs={"lstm_hidden_size": 64})
    assert rec["lstm_hidden_size"] == 64 and rec["n_lstm_layers"] == 1


def test_qrdqn_kwargs_defaults_and_override(capsys):
    cfg = {"algo": "qrdqn", "ppo": {"gamma": 0.97}, "qrdqn": {"buffer_size": 1234}}
    kwargs = build_algo_kwargs(cfg, resolve_algo(cfg), obs_shape_image=(6, 48, 48))
    assert kwargs["buffer_size"] == 1234
    assert kwargs["gamma"] == 0.97  # inherited so it stays aligned with reward.shaping_gamma
    assert "Replay-Buffer" in capsys.readouterr().out  # the RAM estimate must be printed


# ----------------------------------------------------- learn smoke + load dispatch


@pytest.mark.parametrize("algo", ALGO_CHOICES)
def test_learn_smoke_and_load_dispatch(tmp_path: Path, algo: str):
    env = _tiny_vec_env()
    model = _tiny_model(algo, env)
    model.learn(total_timesteps=48)
    path = tmp_path / f"{algo}.zip"
    model.save(str(path))

    assert sniff_algo_name(path) == algo
    loaded = load_trained(path, device="cpu")
    assert type(loaded).__name__ == type(model).__name__

    # jeder Algorithmus muss die einheitliche stateful-predict-Signatur bedienen
    obs = env.reset()
    single = {k: v[0] for k, v in obs.items()}
    state = None
    episode_start = np.ones((1,), dtype=bool)
    for _ in range(3):
        action, state = loaded.predict(single, state=state, episode_start=episode_start, deterministic=True)
        episode_start = np.zeros((1,), dtype=bool)
        assert int(np.asarray(action).reshape(-1)[0]) in (0, 1, 2)
    env.close()


def test_frozen_policy_controller_loads_recurrent(tmp_path: Path):
    from ai.core.env.opponents import FrozenPolicyController

    env = _tiny_vec_env()
    model = _tiny_model("recurrent_ppo", env)
    path = tmp_path / "rec.zip"
    model.save(str(path))
    env.close()

    obs_cfg = ObsConfig(obs_resolution=48, frame_stack=2, n_rays=4, ray_range_px=32.0, arc_horizon=12)
    ctrl = FrozenPolicyController(str(path), obs_cfg)
    play_env = CurveEnv(lambda: EpisodeConfig(opponents=[], enabled_items=set()), config=_tiny_env_config(), seed=1)
    play_env.reset()
    ctrl.reset("fred")
    for _ in range(3):
        a = ctrl.act(play_env.engine, play_env.hero_name, None)
        assert a in (0, 1, 2)
        play_env.step(1)


# ------------------------------------------------------------------- BC mini run


def test_bc_end_to_end(tmp_path: Path, capsys):
    from ai.v1_0.training.bc_pretrain import evaluate_cloned, generate_dataset, train_bc

    cfg = {
        "phase": 1,
        "algo": "ppo",
        "engine_resolution": 256,
        "obs_resolution": 48,
        "frame_stack": 2,
        "n_rays": 4,
        "ray_range_px": 32.0,
        "arc_horizon": 12,
        "max_episode_ticks": 150,
        "cnn_arch": "small",
        "ppo": {"n_steps": 32, "batch_size": 32, "n_epochs": 1, "gamma": 0.99},
        "reward": {"alive_bonus": 0.01, "death_penalty": -1.0},
        "curriculum": {"stages": [{"name": "solo", "n_opponents": 0, "opponent_mix": [], "items_enabled": False}]},
    }
    data_dir = tmp_path / "data"
    meta = generate_dataset(cfg, data_dir, n_transitions=400, teacher="medium", explore_eps=0.1, seed=3)
    assert meta["n_transitions"] == 400
    assert (data_dir / "images.npy").exists()
    assert json.loads((data_dir / "meta.json").read_text())["teacher"] == "medium"
    returns = np.load(data_dir / "returns.npy")
    assert np.isfinite(returns).all() and returns.std() > 0  # echte diskontierte Returns, keine Konstante

    out = tmp_path / "bc_model.zip"
    train_bc(cfg, data_dir, out, epochs=1, batch_size=64, lr=1e-3, seed=3)
    assert out.exists()

    loaded = load_trained(out, device="cpu")
    assert type(loaded).__name__ == "PPO"  # voller SB3-Zip -> direkt --init-from-tauglich

    mean_ticks = evaluate_cloned(cfg, out, n_episodes=2, seed=5)
    assert mean_ticks >= 1

    printed = capsys.readouterr().out
    assert "Trefferquote" in printed
