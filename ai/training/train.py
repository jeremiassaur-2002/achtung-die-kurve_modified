"""Main training entrypoint. One call = one phase, driven entirely by a YAML
config (ai/config/phase{1..5}.yaml) - env/PPO hyperparameters, curriculum stages,
and which of self-play/league to enable. Chain phases with `--init-from` to warm-
start from the previous phase's final checkpoint.

    python -m ai.training.train --config ai/config/phase1.yaml
    python -m ai.training.train --config ai/config/phase2.yaml --init-from ai/runs/phase1_.../final_model.zip
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import set_random_seed

from ai.config import game_constants as gc
from ai.config.game_constants import GameConstants
from ai.env.curve_env import CurveEnv, CurveEnvConfig, RewardConfig
from ai.env.observation import ObsConfig
from ai.env.opponents import OpponentSpec
from ai.env.vec_factory import build_vec_env
from ai.models.policy import POLICY_NAME, build_policy_kwargs
from ai.training.callbacks import (
    CurriculumCallback,
    EvalEloCallback,
    LeagueCallback,
    MetricsLoggingCallback,
    MilestoneReportCallback,
    SelfPlaySnapshotCallback,
)
from ai.training.curriculum import CurriculumManager, CurriculumStage
from ai.training.drive_callbacks import RotatingCheckpointCallback, VideoCallback
from ai.training.run_persistence import latest_checkpoint
from ai.training.league import League
from ai.training.self_play import SelfPlayPool


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _resolve_token(token: str, rng: random.Random, self_play_pool: SelfPlayPool | None, league: League | None) -> OpponentSpec:
    if token == "random":
        return OpponentSpec(kind="random")
    if token == "self_play":
        if self_play_pool is None:
            return OpponentSpec(kind="random")
        return self_play_pool.sample_specs(1, rng)[0]
    if token == "league":
        if league is None:
            return OpponentSpec(kind="random")
        return league.sample_opponent_specs(1, rng)[0]
    if token.startswith("rule_based:"):
        return OpponentSpec(kind="rule_based", difficulty=token.split(":", 1)[1])
    raise ValueError(f"unknown opponent_mix token {token!r}")


def build_stage(stage_cfg: dict, self_play_pool: SelfPlayPool | None, league: League | None) -> CurriculumStage:
    tokens = list(stage_cfg.get("opponent_mix", []))

    def specs_fn(rng: random.Random) -> list[OpponentSpec]:
        return [_resolve_token(t, rng, self_play_pool, league) for t in tokens]

    enabled_items = set(gc.ALL_ITEMS) if stage_cfg.get("items_enabled", False) else set()
    return CurriculumStage(
        name=stage_cfg["name"],
        n_opponents=stage_cfg.get("n_opponents", len(tokens)),
        opponent_specs_fn=specs_fn,
        enabled_items=enabled_items,
        win_rate_threshold=stage_cfg.get("win_rate_threshold", 0.6),
        min_episodes=stage_cfg.get("min_episodes", 200),
    )


def run_training(
    config_path: str | Path,
    init_from: str | None = None,
    resume: str | None = None,
    run_root: str | Path = "ai/runs",
    run_name: str | None = None,
    timesteps_override: int | None = None,
    n_envs_override: int | None = None,
) -> Path:
    cfg = load_config(config_path)
    if n_envs_override is not None:
        cfg["n_envs"] = n_envs_override
    phase = cfg["phase"]
    seed = cfg.get("seed", 0)
    run_name = run_name or f"phase{phase}_{int(time.time())}"
    run_dir = Path(run_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg))

    set_random_seed(seed)

    obs_cfg = ObsConfig(obs_resolution=cfg["obs_resolution"], frame_stack=cfg["frame_stack"])
    reward_cfg = RewardConfig(**cfg.get("reward", {}))
    env_config = CurveEnvConfig(
        engine_resolution=cfg["engine_resolution"],
        obs=obs_cfg,
        enabled_items=set(),
        use_action_masking=cfg.get("use_action_masking", False),
        action_mask_horizon=cfg.get("action_mask_horizon", 3),
        max_episode_ticks=cfg.get("max_episode_ticks", 3600),
        reward=reward_cfg,
    )
    constants = GameConstants(cfg["engine_resolution"])

    self_play_pool = None
    if cfg.get("enable_self_play_snapshots", False):
        self_play_pool = SelfPlayPool(run_dir / "self_play_pool", max_snapshots=cfg.get("self_play_max_snapshots", 10))

    league = None
    if cfg.get("enable_league", False):
        league = League(run_dir / "league")

    curriculum_cfg = cfg["curriculum"]
    stages = [build_stage(s, self_play_pool, league) for s in curriculum_cfg["stages"]]
    manager = CurriculumManager(
        stages,
        blend_window_episodes=curriculum_cfg.get("blend_window_episodes", 100),
        rolling_window=curriculum_cfg.get("rolling_window", 200),
        pool_variants=curriculum_cfg.get("pool_variants", 8),
        rng_seed=seed,
    )
    initial_factory = manager.make_factory(obs_cfg, rng_seed=seed)

    vec_env = build_vec_env(
        n_envs=cfg["n_envs"],
        opponent_factory=initial_factory,
        config=env_config,
        base_seed=seed,
        use_subprocess=cfg.get("use_subprocess", True),
        monitor_dir=str(run_dir / "monitor"),
    )

    use_masking = cfg.get("use_action_masking", False)
    if use_masking:
        from sb3_contrib import MaskablePPO as AlgoCls
    else:
        from stable_baselines3 import PPO as AlgoCls

    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_log = str(tb_dir)
    ppo_kwargs = dict(cfg.get("ppo", {}))
    policy_kwargs = build_policy_kwargs(cnn_arch=cfg.get("cnn_arch", "small"))

    # --resume continues THIS phase's own run (weights + optimizer + step counter,
    # so `total_timesteps` keeps counting where it left off); --init-from only
    # warm-starts weights for a NEW phase. A found resume checkpoint wins over
    # --init-from, which makes the Colab cells idempotent: crash -> just re-run.
    resume_ckpt = None
    if resume == "auto":
        resume_ckpt = latest_checkpoint(run_dir / "checkpoints")
    elif resume:
        resume_ckpt = Path(resume)
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_ckpt}")
    resumed = resume_ckpt is not None

    if resumed:
        model = AlgoCls.load(resume_ckpt, env=vec_env, tensorboard_log=tb_log)
        print(f"[resume] continuing from {resume_ckpt} at {model.num_timesteps:,} steps")
    elif init_from:
        model = AlgoCls.load(init_from, env=vec_env, tensorboard_log=tb_log)
    else:
        model = AlgoCls(POLICY_NAME, vec_env, policy_kwargs=policy_kwargs, tensorboard_log=tb_log, verbose=1, seed=seed, **ppo_kwargs)

    callbacks = [
        # Colab sessions like to die mid-run - without periodic checkpoints everything
        # is lost, because final_model.zip is only written after learn() completes.
        # Rotating: only the newest checkpoint survives (new file fully written first).
        RotatingCheckpointCallback(
            run_dir / "checkpoints",
            every_steps=cfg.get("checkpoint_every_steps", 100_000),
            keep=cfg.get("checkpoint_keep", 1),
            verbose=1,
        ),
        CurriculumCallback(manager, obs_cfg, verbose=1),
        MetricsLoggingCallback(run_dir / "metrics"),
        MilestoneReportCallback(run_dir / "metrics" / "metrics.csv", run_dir / "reports", run_config=cfg, league=league),
    ]
    if self_play_pool is not None:
        callbacks.append(
            SelfPlaySnapshotCallback(
                self_play_pool, manager, obs_cfg, save_every_steps=cfg.get("self_play_snapshot_every_steps", 25_000), verbose=1
            )
        )
    if league is not None:
        callbacks.append(
            LeagueCallback(
                league,
                obs_cfg,
                run_dir / "league_tmp",
                save_every_steps=cfg.get("league_save_every_steps", 50_000),
                n_opponents=cfg.get("league_n_opponents", 5),
                rng_seed=seed,
                verbose=1,
            )
        )
        callbacks.append(
            EvalEloCallback(
                league,
                obs_cfg,
                constants=constants,
                eval_every_steps=cfg.get("league_eval_every_steps", 50_000),
                n_matches=cfg.get("league_eval_matches", 6),
                verbose=1,
            )
        )

    video_every = cfg.get("video_every_steps", 500_000)
    if video_every:
        # one deterministic episode per interval, rendered headlessly to MP4 - the
        # training envs themselves never render (video_every_steps: 0 disables this)
        def _make_video_env() -> CurveEnv:
            return CurveEnv(manager.make_factory(obs_cfg, rng_seed=int(time.time())), config=env_config)

        callbacks.append(
            VideoCallback(
                _make_video_env,
                run_dir / "videos",
                every_steps=video_every,
                max_ticks=int(cfg.get("video_seconds", 30) * 60),
                fps=60,
                verbose=1,
            )
        )

    total_timesteps = timesteps_override or cfg["total_timesteps"]
    # progress_bar=True: rollout collection can be quiet for a while on a CPU-constrained
    # runtime (SB3 only logs metrics once per full rollout, i.e. every n_steps * n_envs
    # steps) - a live bar makes clear it's working rather than looking hung.
    model.learn(
        total_timesteps=total_timesteps, callback=CallbackList(callbacks), progress_bar=True, reset_num_timesteps=not resumed
    )

    final_path = run_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"[train] phase {phase} done -> {final_path}")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Achtung-die-Kurve PPO agent for one curriculum phase.")
    parser.add_argument("--config", required=True, help="path to a phaseN.yaml config")
    parser.add_argument("--init-from", default=None, help="warm-start from a previous phase's final_model.zip")
    parser.add_argument(
        "--resume",
        default=None,
        help="'auto' = continue this run from its newest rotating checkpoint (if any); or an explicit checkpoint path",
    )
    parser.add_argument("--run-root", default="ai/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--timesteps", type=int, default=None, help="override total_timesteps from the config (e.g. for a quick smoke test)")
    parser.add_argument("--n-envs", type=int, default=None, help="override n_envs from the config (match your runtime's actual CPU count - `!nproc` on Colab)")
    args = parser.parse_args()
    run_training(
        args.config,
        init_from=args.init_from,
        resume=args.resume,
        run_root=args.run_root,
        run_name=args.run_name,
        timesteps_override=args.timesteps,
        n_envs_override=args.n_envs,
    )


if __name__ == "__main__":
    main()
