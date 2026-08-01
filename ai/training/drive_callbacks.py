"""SB3 callbacks for (a) rotating, resumable checkpoints and (b) periodic
behavior videos. Both write under the run directory - point --run-root at a
Drive-mounted path (the Colab notebook already does) and everything lands on
Google Drive automatically. The heavy lifting lives in run_persistence.py,
which is torch-free and unit-tested; these classes are just the SB3 glue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

from ai.training.run_persistence import BestTracker, record_episode_video, save_rotating_checkpoint


class BestModelCallback(BaseCallback):
    """Hält zusätzlich zu den rotierenden Checkpoints (= neuester Stand) den
    BESTEN Stand des Laufs fest: gleitender Mittelwert über die letzten `window`
    Episoden von `metric` ("ep_len_mean" = Überlebenszeit in Ticks, oder
    "reward_mean"); verbessert er sich, wird best/best_model.zip atomar ersetzt
    (Logik + Resume-Sicherheit: BestTracker in run_persistence.py). Das ist ein
    vollwertiges model.save() - es lässt sich mit --resume fortsetzen, per
    --init-from in die nächste Phase mitnehmen und direkt exportieren."""

    def __init__(
        self,
        best_dir,
        metric: str = "ep_len_mean",
        window: int = 100,
        min_episodes: int = 50,
        check_every_steps: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.tracker = BestTracker(best_dir, metric=metric, min_episodes=min_episodes)
        self.metric = metric
        self.check_every_steps = check_every_steps
        self._values: deque[float] = deque(maxlen=window)
        self._last_check = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_ticks" not in info:
                continue
            if self.metric == "reward_mean":
                ep = info.get("episode")
                if ep is not None:
                    self._values.append(float(ep["r"]))
            else:
                self._values.append(float(info["episode_ticks"]))

        if self.num_timesteps - self._last_check < self.check_every_steps:
            return True
        self._last_check = self.num_timesteps
        if not self._values:
            return True
        mean_value = sum(self._values) / len(self._values)
        saved = self.tracker.maybe_save(self.model, mean_value, len(self._values), self.num_timesteps)
        if saved is not None and self.verbose:
            print(f"[best] neuer Bestwert {self.metric}={mean_value:.1f} bei Schritt {self.num_timesteps:,} -> {saved}")
        return True


class RotatingCheckpointCallback(BaseCallback):
    """Every `every_steps`, saves a fully resumable checkpoint (weights + optimizer
    + step counter - a plain model.save()) and deletes older ones so only the
    newest `keep` remain. The new file is completely written before the old one
    is removed. Resume with `train.py --resume auto` (or an explicit path)."""

    def __init__(self, ckpt_dir: str | Path, every_steps: int = 100_000, keep: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.ckpt_dir = Path(ckpt_dir)
        self.every_steps = every_steps
        self.keep = keep
        self._last_save = 0

    def _init_callback(self) -> None:
        # resume-aware: after --resume the counter starts at the loaded step count,
        # so don't immediately re-save what we just loaded
        self._last_save = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save < self.every_steps:
            return True
        self._last_save = self.num_timesteps
        path = save_rotating_checkpoint(self.model, self.ckpt_dir, self.num_timesteps, keep=self.keep)
        if self.verbose:
            print(f"[ckpt] saved {path} (keeping newest {self.keep})")
        return True


class VideoCallback(BaseCallback):
    """Every `every_steps`, plays ONE deterministic episode in a fresh env (built
    by `make_env`, so it reflects the curriculum's *current* stage) and writes it
    as MP4 into `video_dir`. Rendering happens only here - the training envs
    themselves stay completely headless, so this costs one episode of sim time
    per `every_steps` and nothing else. Videos are kept (not rotated): they're
    small and the step-by-step progression is the whole point."""

    def __init__(
        self,
        make_env: Callable,
        video_dir: str | Path,
        every_steps: int = 500_000,
        max_ticks: int = 1800,
        fps: int = 60,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.make_env = make_env
        self.video_dir = Path(video_dir)
        self.every_steps = every_steps
        self.max_ticks = max_ticks
        self.fps = fps
        self._last_video = 0

    def _init_callback(self) -> None:
        self._last_video = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_video < self.every_steps:
            return True
        self._last_video = self.num_timesteps
        env = None
        try:
            env = self.make_env()

            def predict(obs):
                action, _ = self.model.predict(obs, deterministic=True)
                return action

            path = record_episode_video(
                env, predict, self.video_dir / f"step_{self.num_timesteps}.mp4", max_ticks=self.max_ticks, fps=self.fps
            )
            if self.verbose:
                print(f"[video] wrote {path}")
        except Exception as exc:  # a broken video must never kill a multi-hour training run
            print(f"[video] recording failed (training continues): {exc!r}")
        finally:
            if env is not None:
                env.close()
        return True
