"""SB3 training callbacks: curriculum advancement, league checkpointing/Elo,
CSV+JSON metric logging (alongside SB3's own TensorBoard logging), and automatic
milestone reports. All read episode outcomes off `infos` (the dicts CurveEnv.step()
returns on termination/truncation - see ai/env/curve_env.py), so none of this needs
CurveEnv to know callbacks exist.
"""

from __future__ import annotations

import csv
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ai.env.observation import ObsConfig
from ai.env.opponents import SpecOpponentFactory
from ai.training.curriculum import CurriculumManager
from ai.training.league import League

MILESTONES = (10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000)


def _finished_episode_infos(infos: list[dict]) -> list[dict]:
    return [info for info in infos if "episode_ticks" in info]


class CurriculumCallback(BaseCallback):
    """Advances CurriculumManager from episode outcomes and pushes a fresh,
    picklable opponent-factory snapshot to every worker whenever composition
    changes (see curriculum.py for why this can't just mutate a shared object)."""

    def __init__(self, manager: CurriculumManager, obs_cfg: ObsConfig, verbose: int = 0):
        super().__init__(verbose)
        self.manager = manager
        self.obs_cfg = obs_cfg

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        changed = False
        for info in _finished_episode_infos(infos):
            if self.manager.record_episode(bool(info.get("won", False))):
                changed = True

        if changed:
            factory = self.manager.make_factory(self.obs_cfg, rng_seed=self.num_timesteps)
            self.training_env.env_method("set_opponent_factory", factory)
            if self.verbose:
                print(f"[curriculum] -> stage={self.manager.current_stage.name} win_rate={self.manager.win_rate():.2f}")

        self.logger.record("curriculum/stage_idx", self.manager.stage_idx)
        self.logger.record("curriculum/win_rate", self.manager.win_rate())
        return True


class LeagueCallback(BaseCallback):
    """Saves a checkpoint into the League every `save_every_steps` and refreshes
    the training opponent mix (current + historical + rule-based + random) - Phase 4."""

    def __init__(
        self,
        league: League,
        obs_cfg: ObsConfig,
        tmp_dir: str | Path,
        save_every_steps: int = 50_000,
        n_opponents: int = 3,
        rng_seed: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.league = league
        self.obs_cfg = obs_cfg
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_steps = save_every_steps
        self.n_opponents = n_opponents
        self.rng = random.Random(rng_seed)
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save < self.save_every_steps:
            return True
        self._last_save = self.num_timesteps

        tmp_path = self.tmp_dir / f"tmp_{self.num_timesteps}.zip"
        self.model.save(str(tmp_path))
        entry = self.league.add_checkpoint(tmp_path, self.num_timesteps)
        tmp_path.unlink(missing_ok=True)

        specs = self.league.sample_opponent_specs(self.n_opponents, self.rng, current_model_path=entry.path)
        factory = SpecOpponentFactory(specs, self.obs_cfg, rng_seed=self.num_timesteps)
        self.training_env.env_method("set_opponent_factory", factory)
        if self.verbose:
            print(f"[league] checkpoint {entry.name} saved at step {self.num_timesteps}")
        return True


class EvalEloCallback(BaseCallback):
    """Periodically runs real arena matches (not training rollouts) between the
    current model and league/rule-based/random opponents, and feeds the placement
    order into League Elo - this is what makes Elo an objective measurement rather
    than a guess from training reward alone."""

    def __init__(
        self,
        league: League,
        obs_cfg: ObsConfig,
        constants=None,
        eval_every_steps: int = 50_000,
        n_matches: int = 6,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.league = league
        self.obs_cfg = obs_cfg
        self.constants = constants
        self.eval_every_steps = eval_every_steps
        self.n_matches = n_matches
        self._last_eval = 0
        self._tmp_dir: Path | None = None

    def _on_training_start(self) -> None:
        self._tmp_dir = Path(self.model.logger.dir or ".") / "elo_tmp"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_every_steps:
            return True
        self._last_eval = self.num_timesteps

        from ai.evaluation.arena import run_match
        from ai.env.opponents import FrozenPolicyController, OpponentSpec, build_controller

        assert self._tmp_dir is not None
        tmp_path = self._tmp_dir / f"current_{self.num_timesteps}.zip"
        self.model.save(str(tmp_path))
        rng = random.Random(self.num_timesteps)

        for _ in range(self.n_matches):
            opponent_specs = self.league.sample_opponent_specs(3, rng, current_model_path=None)
            participants = {"current": FrozenPolicyController(str(tmp_path), self.obs_cfg, deterministic=True)}
            for i, spec in enumerate(opponent_specs):
                label = spec.model_path if spec.kind == "frozen" else f"{spec.kind}:{spec.difficulty or i}"
                participants[label] = build_controller(spec, self.obs_cfg, rng)
            result = run_match(participants, constants=self.constants, seed=rng.randrange(2**31))

            placements = [("current" if p == "current" else _league_name_for_path(self.league, p)) for p in result.placements]
            self.league.record_match(placements)

        tmp_path.unlink(missing_ok=True)
        best = self.league.best()
        if best is not None:
            self.logger.record("league/best_elo", self.league.elo.get(best.name))
            self.logger.record("league/best_name", best.name)
        return True


def _league_name_for_path(league: League, path_or_label: str) -> str:
    for entry in league.entries.values():
        if entry.path == path_or_label:
            return entry.name
    return path_or_label  # already a stable label like "rule_based:medium" or "random"


class MetricsLoggingCallback(BaseCallback):
    """Mirrors reward/win-rate/episode-length plus SB3's own loss/entropy/LR values
    to CSV + JSON lines, so ai/reporting/plots.py doesn't need TensorBoard's event
    file format to build charts."""

    def __init__(self, out_dir: str | Path, log_every_steps: int = 2000, rolling_window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "metrics.csv"
        self.jsonl_path = self.out_dir / "metrics.jsonl"
        self.log_every_steps = log_every_steps
        self._last_log = 0
        self._ep_rewards: deque[float] = deque(maxlen=rolling_window)
        self._ep_lengths: deque[float] = deque(maxlen=rolling_window)
        self._wins: deque[float] = deque(maxlen=rolling_window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in _finished_episode_infos(infos):
            self._ep_lengths.append(info["episode_ticks"])
            self._wins.append(1.0 if info.get("won") else 0.0)
            ep = info.get("episode")  # present if envs are wrapped with SB3's Monitor
            if ep is not None:
                self._ep_rewards.append(ep["r"])

        if self.num_timesteps - self._last_log >= self.log_every_steps:
            self._last_log = self.num_timesteps
            self._write_row(self._collect_row())
        return True

    def _collect_row(self) -> dict:
        name_to_value = getattr(self.model.logger, "name_to_value", {}) if hasattr(self.model, "logger") else {}
        return {
            "step": self.num_timesteps,
            "reward_mean": float(np.mean(self._ep_rewards)) if self._ep_rewards else float("nan"),
            "win_rate": float(np.mean(self._wins)) if self._wins else float("nan"),
            "ep_len_mean": float(np.mean(self._ep_lengths)) if self._ep_lengths else float("nan"),
            "policy_loss": name_to_value.get("train/policy_gradient_loss", float("nan")),
            "value_loss": name_to_value.get("train/value_loss", float("nan")),
            "entropy_loss": name_to_value.get("train/entropy_loss", float("nan")),
            "learning_rate": name_to_value.get("train/learning_rate", float("nan")),
        }

    def _write_row(self, row: dict) -> None:
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")


class MilestoneReportCallback(BaseCallback):
    """Fires ai/reporting/report.py once per crossed step milestone."""

    def __init__(self, metrics_csv: str | Path, report_dir: str | Path, run_config: dict, league: League | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.metrics_csv = Path(metrics_csv)
        self.report_dir = Path(report_dir)
        self.run_config = run_config
        self.league = league
        self._done: set[int] = set()

    def _on_step(self) -> bool:
        for milestone in MILESTONES:
            if milestone in self._done or self.num_timesteps < milestone:
                continue
            self._done.add(milestone)
            try:
                from ai.reporting.report import generate_report

                generate_report(self.metrics_csv, self.report_dir, milestone=milestone, run_config=self.run_config, league=self.league)
                if self.verbose:
                    print(f"[report] generated for milestone {milestone}")
            except Exception as e:  # reporting must never crash training
                print(f"[report] failed at milestone {milestone}: {e}")
        return True


class SelfPlaySnapshotCallback(BaseCallback):
    """Periodically saves the model-in-training into the self-play pool and
    refreshes the opponent factory so newer snapshots actually get faced - needed
    because Phase 1/2 style single-stage curricula never fire CurriculumCallback's
    own refresh (there's no stage to advance to), so without this the self-play
    opponent would stay frozen at whatever existed when training started."""

    def __init__(self, pool, manager: CurriculumManager, obs_cfg: ObsConfig, save_every_steps: int = 25_000, verbose: int = 0):
        super().__init__(verbose)
        self.pool = pool
        self.manager = manager
        self.obs_cfg = obs_cfg
        self.save_every_steps = save_every_steps
        self._last_save = 0
        self._tmp_dir: Path | None = None

    def _on_training_start(self) -> None:
        self._tmp_dir = Path(self.model.logger.dir or ".") / "self_play_tmp"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save < self.save_every_steps:
            return True
        self._last_save = self.num_timesteps

        assert self._tmp_dir is not None
        tmp_path = self._tmp_dir / f"tmp_{self.num_timesteps}.zip"
        self.model.save(str(tmp_path))
        self.pool.add_snapshot(tmp_path, self.num_timesteps)
        tmp_path.unlink(missing_ok=True)

        factory = self.manager.make_factory(self.obs_cfg, rng_seed=self.num_timesteps)
        self.training_env.env_method("set_opponent_factory", factory)
        if self.verbose:
            print(f"[self-play] snapshot added at step {self.num_timesteps}")
        return True
