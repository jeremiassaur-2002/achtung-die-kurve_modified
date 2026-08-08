"""Torch-frei: BestTracker (beste Gewichte), Video-Ende-Erkennung inkl. Sidecar,
die neuen Report-Charts (Überlebenszeit, mean_-Varianten, Todesursachen) und die
Report-CLI zum Nachgenerieren verpasster Meilensteine."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai.core.env.curve_env import CurveEnv, CurveEnvConfig
from ai.core.env.observation import ObsConfig
from ai.core.env.opponents import EpisodeConfig
from ai.v1_0.reporting import plots, report
from ai.v1_0.training.run_persistence import BestTracker, record_episode_video


class _StubModel:
    def __init__(self):
        self.saved_to: list[str] = []

    def save(self, path: str) -> None:
        self.saved_to.append(path)
        Path(path).write_bytes(b"fake-weights")


def _solo_factory() -> EpisodeConfig:
    return EpisodeConfig(opponents=[], enabled_items=set())


def _tiny_env(max_episode_ticks: int) -> CurveEnv:
    cfg = CurveEnvConfig(engine_resolution=64, obs=ObsConfig(obs_resolution=32, frame_stack=2), max_episode_ticks=max_episode_ticks)
    return CurveEnv(_solo_factory, config=cfg, seed=3)


# --------------------------------------------------------------- BestTracker


def test_best_tracker_burn_in_and_improvement(tmp_path):
    tracker = BestTracker(tmp_path / "best", metric="ep_len_mean", min_episodes=10)
    model = _StubModel()

    assert tracker.maybe_save(model, mean_value=500.0, n_episodes=3, step=1000) is None  # Burn-in
    assert tracker.maybe_save(model, mean_value=500.0, n_episodes=50, step=2000) is not None
    assert tracker.maybe_save(model, mean_value=400.0, n_episodes=50, step=3000) is None  # schlechter
    assert tracker.maybe_save(model, mean_value=600.0, n_episodes=50, step=4000) is not None

    meta = json.loads((tmp_path / "best" / "best_model.json").read_text())
    assert meta["value"] == 600.0 and meta["step"] == 4000
    history = (tmp_path / "best" / "best_history.csv").read_text().strip().splitlines()
    assert len(history) == 1 + 2  # Header + zwei Verbesserungen
    assert (tmp_path / "best" / "best_model.zip").read_bytes() == b"fake-weights"


def test_best_tracker_resume_keeps_historic_best(tmp_path):
    best_dir = tmp_path / "best"
    BestTracker(best_dir, min_episodes=1).maybe_save(_StubModel(), 900.0, 50, 5000)

    resumed = BestTracker(best_dir, min_episodes=1)  # wie nach --resume: frisches Objekt
    assert resumed.best_value == 900.0
    assert resumed.maybe_save(_StubModel(), 800.0, 50, 6000) is None  # darf 900 nicht überschreiben
    assert resumed.maybe_save(_StubModel(), 950.0, 50, 7000) is not None


# ------------------------------------------------------- Video-Ende-Erkennung


def test_video_recording_limit_writes_survivor_sidecar(tmp_path):
    env = _tiny_env(max_episode_ticks=400)
    out = record_episode_video(env, lambda obs: 1, tmp_path / "clip.mp4", max_ticks=3, fps=10)
    sidecar = json.loads(Path(out).with_suffix(".json").read_text())
    assert sidecar["end_reason"] == "recording_limit"  # Agent lebt noch - genau der Fall der 30s-Videos
    assert sidecar["death_cause"] is None
    assert sidecar["ticks_survived"] == 3
    env.close()


def test_video_until_episode_end_names_reason(tmp_path):
    env = _tiny_env(max_episode_ticks=200)
    rng = np.random.default_rng(0)
    out = record_episode_video(env, lambda obs: int(rng.integers(0, 3)), tmp_path / "clip.mp4", max_ticks=1000, fps=10)
    sidecar = json.loads(Path(out).with_suffix(".json").read_text())
    assert sidecar["end_reason"] in ("death", "episode_limit")  # Aufnahme deckt die ganze Episode ab
    if sidecar["end_reason"] == "death":
        assert sidecar["death_cause"] is not None
    assert Path(out).exists()
    env.close()


# ----------------------------------------------------------------- Charts/CLI


def _synthetic_metrics(n_rows: int = 3000) -> pd.DataFrame:
    steps = np.arange(1, n_rows + 1) * 500
    noise = np.random.default_rng(1).normal(0, 3, n_rows)
    return pd.DataFrame(
        {
            "step": steps,
            "reward_mean": 10 + 5 * np.sin(steps / 2e5) + noise,
            "win_rate": np.zeros(n_rows),
            "ep_len_mean": 900 + 400 * np.sin(steps / 3e5) + noise * 30,
            "policy_loss": noise / 10,
            "value_loss": np.abs(noise),
            "entropy_loss": -0.9 + noise / 100,
            "learning_rate": np.full(n_rows, 3e-4),
        }
    )


def test_survival_and_mean_charts(tmp_path):
    metrics = _synthetic_metrics()
    assert plots.survival_time_curve(metrics, tmp_path).exists()
    written = plots.mean_variants(metrics, tmp_path)
    names = {p.name for _t, p in written}
    assert {"mean_reward_curve.png", "mean_survival_time_curve.png", "mean_entropy_curve.png"} <= names
    # fehlende Spalte crasht nicht, sie fällt einfach weg
    partial = plots.mean_variants(metrics.drop(columns=["entropy_loss"]), tmp_path / "sub")
    assert "mean_entropy_curve.png" not in {p.name for _t, p in partial}


def test_death_causes_curve(tmp_path):
    dc = pd.DataFrame({"step": [1000, 2000, 3000], "border": [0.5, 0.4, 0.2], "self": [0.5, 0.5, 0.3], "doomed": [0.0, 0.1, 0.5]})
    assert plots.death_causes_curve(dc, tmp_path).exists()


def test_report_cli_generates_missing_milestone(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "phase1"
    (run_dir / "metrics").mkdir(parents=True)
    _synthetic_metrics(400).to_csv(run_dir / "metrics" / "metrics.csv", index=False)
    (run_dir / "config_used.yaml").write_text("phase: 1\ncurriculum:\n  stages:\n    - name: solo\n      n_opponents: 0\n")

    monkeypatch.setattr(
        "sys.argv", ["report", "--run-dir", str(run_dir), "--milestones", "2000000", "3000000"]
    )
    report.main()

    for milestone in (2000000, 3000000):
        md = run_dir / "reports" / f"milestone_{milestone}" / "report.md"
        assert md.exists()
        text = md.read_text(encoding="utf-8")
        assert "Überlebenszeit" in text and "strukturell immer 0%" in text
        assert (md.parent / "mean_reward_curve.png").exists()
        assert (md.parent / "survival_time_curve.png").exists()

    report.main()  # zweiter Lauf: vorhandene Meilensteine werden übersprungen, kein Crash
    assert "übersprungen" in capsys.readouterr().out
