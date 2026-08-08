"""Tests fuer die Versionsaufteilung: gemeinsamer Kern, Planer, Timing, Entrypoint."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from ai.core.config.game_constants import GameConstants
from ai.core.env import sensors
from ai.core.env.engine import CurveEngine, STRAIGHT, TURN_LEFT, TURN_RIGHT
from ai.core.utils.paths import run_paths
from ai.core.utils.timing import PhaseTimer
from ai.v1_1.planner.beam import BeamPlanner, PlannerConfig, PlannerController


def _engine(seed: int = 0, ticks: int = 0, players=("fred", "greenlee")) -> CurveEngine:
    eng = CurveEngine(GameConstants(256))
    eng.reset(list(players), seed=seed)
    rng = random.Random(seed)
    for _ in range(ticks):
        if not eng.players["fred"].alive:
            break
        eng.step({n: rng.choice([TURN_LEFT, STRAIGHT, TURN_RIGHT]) for n in eng.players})
    return eng


# --------------------------------------------------------------- Struktur


def test_core_imports_without_any_version_package():
    """Der Kern muss sich importieren lassen, wenn ai.v1_0 und ai.v1_1 gar nicht
    existieren - sonst waere der Schnitt zwischen Kern und Version nur optisch.

    Geprueft wird im Subprozess mit blockierten Versionspaketen, nicht per
    Textsuche: entscheidend ist, ob ein Import beim LADEN passiert. Die beiden
    verbliebenen Rueckgriffe auf v1_0 (FrozenPolicyController laedt ein
    SB3-Modell, evaluate.py kennt die League) stehen bewusst INNERHALB ihrer
    Funktionen - wer sie aufruft, braucht v1_0, wer nur die Engine nutzt, nicht.
    Wandert einer davon nach oben, faellt dieser Test.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys, importlib, pkgutil
        class Block:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)
            def find_spec(self, name, path=None, target=None):
                if name.startswith(("ai.v1_0", "ai.v1_1")):
                    raise ImportError("Versionspaket im Kern nicht erlaubt: " + name)
                return None
        sys.meta_path.insert(0, Block())
        import ai.core
        failed = []
        for m in pkgutil.walk_packages(ai.core.__path__, "ai.core."):
            try:
                importlib.import_module(m.name)
            except ImportError as e:
                failed.append(f"{m.name}: {e}")
        print("FAILED:" + "|".join(failed))
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    failures = out.stdout.split("FAILED:")[-1].strip()
    assert not failures, f"ai/core laesst sich ohne Versionspakete nicht importieren: {failures}"


# ------------------------------------------------------- simulate_deltas


@pytest.mark.parametrize("seed", [0, 3, 7, 11])
def test_simulate_deltas_matches_arc_survival(seed):
    """Die reinen Strategien durch die allgemeine Simulation gejagt muessen exakt
    das liefern, was arc_survival_ticks sagt - sonst haben Planer und
    Doom-Terminator zwei verschiedene Weltbilder."""
    eng = _engine(seed=seed, ticks=40)
    if not eng.players["fred"].alive:
        pytest.skip("Held bereits tot")
    horizon = 45
    arc = sensors.arc_survival_ticks(eng, "fred", horizon)
    deltas = np.array([[-1.0] * horizon, [0.0] * horizon, [1.0] * horizon], dtype=np.float32)
    _, survived = sensors.simulate_deltas(eng, "fred", deltas)
    assert np.array_equal(arc, survived)


def test_simulate_deltas_shapes_and_empty():
    eng = _engine(seed=1)
    pos, surv = sensors.simulate_deltas(eng, "fred", np.zeros((5, 12), dtype=np.float32))
    assert pos.shape == (5, 12, 2) and surv.shape == (5,)
    pos0, surv0 = sensors.simulate_deltas(eng, "fred", np.zeros((3, 0), dtype=np.float32))
    assert pos0.shape == (3, 0, 2) and surv0.tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        sensors.simulate_deltas(eng, "fred", np.zeros(7, dtype=np.float32))


def test_clearance_is_low_in_corner_and_high_in_center():
    eng = _engine(seed=2)
    c = eng.c
    b = c.border_width + c.hitbox_size + eng.field_inset
    pts = np.array([[b - 1.0, b - 1.0], [128.0, 128.0]], dtype=np.float32)
    vals = sensors.clearance_at(eng, "fred", pts, n_rays=12, range_px=48.0)
    assert vals[0] < 0.1, "Ecke muesste als eng gelten"
    assert vals[1] > 0.9, "Feldmitte muesste als frei gelten"
    assert np.all((vals >= 0.0) & (vals <= 1.0))


# ------------------------------------------------------------- Planer


def test_planner_returns_valid_action():
    eng = _engine(seed=4, ticks=30)
    if not eng.players["fred"].alive:
        pytest.skip("Held bereits tot")
    res = BeamPlanner(PlannerConfig(horizon=24, commit_ticks=6, beam_width=4)).plan(eng, "fred")
    assert res.action in (TURN_LEFT, STRAIGHT, TURN_RIGHT)
    assert 0 <= res.survived <= 24
    assert res.n_evaluated > 0


def test_planner_beats_random_survival():
    """Der Planer soll deutlich laenger ueberleben als Zufall. Ohne diese
    Schranke koennte ein kaputter Score-Term unbemerkt bleiben - die Suche
    liefert ja weiterhin gueltige Aktionen."""
    # horizon=60 (1 Sekunde) ist der Default und der fachlich richtige Wert: der
    # Wendekreisradius betraegt v/omega ~ 10 px, eine Halbdrehung dauert ~52
    # Ticks. Mit einem kuerzeren Horizont kann die Suche gar nicht erkennen,
    # dass sie in eine Spirale faehrt.
    cfg = PlannerConfig(horizon=60, commit_ticks=6, beam_width=6)

    def survive(decide, seed):
        eng = CurveEngine(GameConstants(256))
        eng.reset(["fred"], seed=seed)
        t = 0
        while eng.players["fred"].alive and t < 900:
            eng.step({"fred": decide(eng)})
            t += 1
        return t

    planner = BeamPlanner(cfg)
    rng = random.Random(0)
    planned = [survive(lambda e: planner.act(e, "fred"), s) for s in range(3)]
    chaotic = [survive(lambda e: rng.choice([TURN_LEFT, STRAIGHT, TURN_RIGHT]), s) for s in range(3)]
    assert np.mean(planned) > 3 * np.mean(chaotic), f"Planer {np.mean(planned):.0f} vs Zufall {np.mean(chaotic):.0f}"


def test_planner_avoids_a_certain_wall():
    """Kopf dicht vor der Wand, geradeaus toedlich: der Planer darf nicht
    geradeaus waehlen."""
    eng = _engine(seed=5)
    p = eng.players["fred"]
    b = eng.c.border_width + eng.c.hitbox_size + eng.field_inset
    # 25 px Abstand: der Wendekreisradius ist v/omega ~ 10 px, eine Drehung von
    # senkrecht auf parallel naehert sich der Wand um r*(1-cos(90 Grad)) = r.
    # 25 > 10, Ausweichen ist also moeglich - bei 3 px waere es das NICHT, und
    # STRAIGHT waere dann korrekt (alle Optionen sterben, der Bonus entscheidet).
    p.x, p.y = 128.0, b + 25.0
    p.dir = -np.pi / 2  # direkt auf die obere Wand zu
    planner = BeamPlanner(PlannerConfig(horizon=60, commit_ticks=5, beam_width=8))
    res = planner.plan(eng, "fred")
    assert res.action != STRAIGHT
    assert res.survived == 60, "Ausweichen muesste den ganzen Horizont ueberleben"


def test_planner_accepts_doom_without_crashing():
    """3 px vor der Wand ist der Aufprall geometrisch unvermeidbar. Der Planer
    muss trotzdem eine gueltige Aktion liefern statt zu werfen - eine verlorene
    Lage ist kein Programmfehler."""
    eng = _engine(seed=5)
    p = eng.players["fred"]
    b = eng.c.border_width + eng.c.hitbox_size + eng.field_inset
    p.x, p.y = 128.0, b + 3.0
    p.dir = -np.pi / 2
    res = BeamPlanner(PlannerConfig(horizon=60, commit_ticks=5, beam_width=8)).plan(eng, "fred")
    assert res.action in (TURN_LEFT, STRAIGHT, TURN_RIGHT)
    assert res.survived < 60


def test_planner_controller_interface():
    eng = _engine(seed=6, ticks=10)
    if not eng.players["fred"].alive:
        pytest.skip("Held bereits tot")
    ctrl = PlannerController(PlannerConfig(horizon=18, commit_ticks=6, beam_width=3))
    ctrl.reset("fred")
    assert ctrl.act(eng, "fred", None) in (TURN_LEFT, STRAIGHT, TURN_RIGHT)


def test_planner_config_rejects_nonsense():
    for bad in (dict(horizon=0), dict(commit_ticks=0), dict(beam_width=0)):
        with pytest.raises(ValueError):
            PlannerConfig(**bad)


# ------------------------------------------------------------- Timing


def test_phase_timer_records_and_writes(tmp_path):
    with PhaseTimer(tmp_path, run_label="t") as timer:
        with timer.phase("setup", note="x"):
            pass
        for _ in range(3):
            with timer.section("step"):
                pass
    data = json.loads((tmp_path / "timing.json").read_text())
    assert [p["name"] for p in data["phases"]] == ["setup"]
    assert data["phases"][0]["meta"] == {"note": "x"}
    step = next(s for s in data["sections"] if s["name"] == "step")
    assert step["calls"] == 3
    md = (tmp_path / "timing.md").read_text()
    assert "setup" in md and "Abschnitte" in md


def test_phase_timer_records_time_even_on_exception(tmp_path):
    """Ausgerechnet der abgestuerzte Abschnitt darf nicht ohne Zeitangabe
    dastehen - sonst fehlt die Information genau dann, wenn man sie braucht."""
    timer = PhaseTimer(tmp_path, run_label="t")
    with pytest.raises(RuntimeError):
        with timer.phase("kaputt"):
            raise RuntimeError("boom")
    data = json.loads((tmp_path / "timing.json").read_text())
    assert data["phases"][0]["name"] == "kaputt"
    assert data["phases"][0]["seconds"] >= 0.0


# -------------------------------------------------------- Pfade / CLI


def test_run_paths_layout(tmp_path):
    p = run_paths("v1_1", "lauf", tmp_path).mkdirs()
    assert p.tensorboard == tmp_path / "v1_1" / "lauf" / "tensorboard"
    for d in (p.checkpoints, p.best, p.videos, p.reports, p.metrics, p.timing):
        assert d.is_dir()


def test_run_cli_rejects_unknown_version():
    from ai.run import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--version", "v9_9", "--config", "x.yaml"])


def test_run_cli_parses_both_versions():
    from ai.run import build_parser

    for v in ("v1_0", "v1_1"):
        args = build_parser().parse_args(["train", "--version", v, "--config", "c.yaml"])
        assert args.version == v and args.run_name == "auto" and args.resume == "auto"


# ------------------------------------------------- 36 Sensoren / Vektor


def test_36_rays_give_10_degree_spacing_and_expected_vector_dim():
    from ai.core.env.observation import ObsConfig, ObservationBuilder, vector_dim

    cfg = ObsConfig(obs_resolution=64, frame_stack=1, n_rays=36, ray_range_px=64.0)
    # 16 Basiswerte + 6 Farb-One-Hot + 36 Strahlen + 3 Arc-Werte
    assert vector_dim(cfg) == 61
    eng = _engine(seed=8, ticks=20)
    if not eng.players["fred"].alive:
        pytest.skip("Held bereits tot")
    obs = ObservationBuilder(cfg).observe(eng, "fred")
    assert obs["vector"].shape == (61,)
    rays = sensors.ray_distances(eng, "fred", 36, 64.0)
    assert rays.shape == (36,) and np.all((rays >= 0.0) & (rays <= 1.0))
    # Strahl k liegt bei 2*pi*k/36 = genau 10 Grad Schrittweite
    units = sensors._ray_units(36)
    angles = np.degrees(np.arctan2(units[:, 1], units[:, 0])) % 360.0
    diffs = np.diff(np.sort(angles))
    assert np.allclose(diffs, 10.0, atol=1e-4)
