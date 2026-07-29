"""Gymnasium API compliance + a couple of tiny end-to-end smoke checks: the env
resets/steps with correct shapes, action masking never masks all three actions,
and a genuinely short PPO training run doesn't crash (CNN extractor, policy,
rollout collection all wired together)."""

import numpy as np

from ai.env.curve_env import CurveEnv, CurveEnvConfig
from ai.env.observation import ObsConfig
from ai.env.opponents import EpisodeConfig, RandomController


def _simple_factory() -> EpisodeConfig:
    return EpisodeConfig(opponents=[RandomController()], enabled_items=set())


def _tiny_config(**overrides) -> CurveEnvConfig:
    defaults = dict(engine_resolution=64, obs=ObsConfig(obs_resolution=32, frame_stack=2), max_episode_ticks=200)
    defaults.update(overrides)
    return CurveEnvConfig(**defaults)


def test_gym_api_compliance():
    from gymnasium.utils.env_checker import check_env

    env = CurveEnv(_simple_factory, config=_tiny_config(), seed=0)
    check_env(env, warn=True, skip_render_check=True)


def test_reset_and_step_shapes():
    env = CurveEnv(_simple_factory, config=_tiny_config(max_episode_ticks=50), seed=1)
    obs, info = env.reset()
    assert obs["image"].shape == (2 * 3, 32, 32)
    assert obs["image"].dtype == np.uint8
    assert obs["vector"].shape[0] > 0

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 500:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
    assert terminated or truncated
    assert "episode_ticks" in info


def test_action_masking_never_masks_all_three():
    env = CurveEnv(_simple_factory, config=_tiny_config(use_action_masking=True), seed=2)
    env.reset()
    for _ in range(50):
        mask = env.action_masks()
        assert mask.any()
        action = int(np.argmax(mask))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()


def test_short_ppo_smoke_train():
    from stable_baselines3 import PPO

    from ai.env.vec_factory import build_vec_env
    from ai.models.policy import POLICY_NAME, build_policy_kwargs

    # SmallCNN's fixed 8/4/4/2/3/1 conv strides need >= ~64px input (32px, used by
    # the other tests here for speed, is smaller than its own receptive field)
    config = _tiny_config(obs=ObsConfig(obs_resolution=64, frame_stack=2), max_episode_ticks=100)
    vec_env = build_vec_env(n_envs=1, opponent_factory=_simple_factory, config=config, use_subprocess=False)
    policy_kwargs = build_policy_kwargs(cnn_arch="small", cnn_features_dim=32, vector_features_dim=16)
    model = PPO(POLICY_NAME, vec_env, policy_kwargs=policy_kwargs, n_steps=64, batch_size=32, n_epochs=1, verbose=0)
    model.learn(total_timesteps=128)
