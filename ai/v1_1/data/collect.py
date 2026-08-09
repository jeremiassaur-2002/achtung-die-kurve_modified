"""Trajektorien mit dem Beam-Planer als Experten sammeln - die Datenbasis fuer
das statische (offline) Training von v1_1.

    python -m ai.v1_1.data.collect --config ai/v1_1/config/dreamer.yaml --episodes 200

**Was gespeichert wird und warum genau das.** Pro Tick ein EINZELNER Frame
(H, W, 3 uint8), NICHT der 4er-Frame-Stack, den v1_0 der CNN fuettert. Zwei
Gruende:

- Groesse. Ein 96er-Stack ist 110 KB pro Tick; eine Million Ticks waeren 110 GB.
  Ohne Stack sind es 27 KB, also ~27 GB - und bei 64x64 (DreamerV3-Standard)
  noch 12 GB. Auf einer gemieteten Maschine mit 50-100 GB Volume ist das der
  Unterschied zwischen "geht" und "geht nicht".
- Redundanz. Ein Frame-Stack ist die Kruecke einer zustandslosen Policy. Der
  RSSM des Weltmodells FUEHRT einen rekurrenten Zustand - er baut sich die
  Historie selbst. Den Stack mitzuspeichern hiesse, dieselbe Information viermal
  auf die Platte zu schreiben.

Der Sensorvektor (inkl. der 36 Strahlen) wird mitgespeichert. Er ist winzig
(~230 B/Tick) und enthaelt genau die Nahfeldgeometrie, die aus einem
heruntergerechneten Bild nicht mehr rekonstruierbar ist.

**Format**: ein `.npz` je Episode in `<out>/shards/`, plus `index.json` mit den
Kennzahlen. Episodenweise statt einer grossen Datei, weil das Sammeln
parallelisierbar sein muss (der Planer schafft ~170 Ticks/s auf einem Kern -
mit `--workers 8` wird daraus eine brauchbare Rate) und ein Abbruch mittendrin
nur die letzte Episode kostet.

**Warum gemischte Experten-Qualitaeten (`planner_mix`).** Ein einheitlich starker
Planer ueberlebt tausende Ticks und stirbt fast nie - der Datensatz enthaelt dann
unter 0,3% Todesschritte, und der Fortsetzungskopf des Weltmodells lernt schlicht
"es geht immer weiter". Der Actor haette in der Imagination nichts zu vermeiden.
Deshalb wird ein Teil der Episoden mit bewusst schwaecheren Einstellungen
gesammelt (kurzer Horizont, schmaler Beam, mehr Rauschen): die sterben oft und
liefern genau die Aufpralle, aus denen das Modell die Konsequenz eines Fehlers
lernt. Die starken Episoden liefern weiterhin das gute Verhalten fuer die
Verhaltensklonung. Beides wird gebraucht.

**Warum Epsilon-Rauschen im Experten.** Ein rein deterministischer Planer stirbt
fast nie und faehrt sehr aehnliche Bahnen. Ein Weltmodell, das daraus lernt,
sieht nie einen Aufprall und kann die Konsequenz eines Fehlers nicht vorhersagen
- und ein daraus geklonter Actor hat keine Ahnung, wie er sich aus einer Lage
befreit, in die der Experte nie geraten ist. `planner.epsilon` (Default 0.05)
streut genug, um beides in den Datensatz zu bekommen.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from ai.core.env import renderer
from ai.core.env.curve_env import CurveEnv, CurveEnvConfig, RewardConfig
from ai.core.env.observation import ObsConfig
from ai.core.env.opponents import EpisodeConfig, OpponentSpec, build_controller
from ai.core.utils.timing import PhaseTimer
from ai.v1_1.planner.beam import BeamPlanner, PlannerConfig


def _episode_factory(obs_cfg: ObsConfig, opponent_tokens: list[str], rng: random.Random):
    """Minimale Gegner-Fabrik: v1_1 braucht (noch) kein Curriculum, nur eine
    reproduzierbare Mischung aus Zufalls- und Regelgegnern."""

    def _make() -> EpisodeConfig:
        controllers = []
        for tok in opponent_tokens:
            if tok == "random":
                spec = OpponentSpec(kind="random")
            elif tok.startswith("rule_based:"):
                spec = OpponentSpec(kind="rule_based", difficulty=tok.split(":", 1)[1])
            else:
                raise ValueError(f"unbekannter opponent-Token {tok!r} (erlaubt: random, rule_based:<stufe>)")
            controllers.append(build_controller(spec, obs_cfg, rng))
        return EpisodeConfig(opponents=controllers)

    return _make


def collect_episode(env: CurveEnv, planner: BeamPlanner, seed: int) -> dict[str, np.ndarray]:
    """Eine Episode; der Planer steuert den Helden, die Gegner ihre Sitze."""
    obs, _ = env.reset(seed=seed)
    frames, vectors, actions, rewards, dones, survived = [], [], [], [], [], []
    engine = env.engine
    hero = env.hero_name
    done = False
    while not done:
        # Der Frame kommt aus dem Renderer statt aus obs["image"], weil obs den
        # gestapelten Verlauf enthaelt - hier soll genau der aktuelle Frame rein.
        frames.append(renderer.render_frame(engine, env.config.obs.obs_resolution))
        vectors.append(np.asarray(obs["vector"], dtype=np.float32))
        plan = planner.plan(engine, hero)
        actions.append(plan.action)
        survived.append(plan.survived)
        obs, reward, terminated, truncated, _ = env.step(plan.action)
        rewards.append(reward)
        done = terminated or truncated
        dones.append(done)
    return {
        "frames": np.stack(frames).astype(np.uint8),
        "vectors": np.stack(vectors).astype(np.float32),
        "actions": np.asarray(actions, dtype=np.int8),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
        # Der Restsurvival-Schaetzer des Planers zum Zeitpunkt der Entscheidung:
        # ein kostenloses Hilfsziel, auf das sich der Critic vortrainieren laesst.
        "planner_survived": np.asarray(survived, dtype=np.int16),
    }


def collect(config_path: str | Path, out_dir: str | Path, episodes: int, seed: int = 0, worker_id: int = 0) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(out_dir)
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)

    obs_cfg = ObsConfig(
        obs_resolution=cfg["obs_resolution"],
        frame_stack=cfg.get("frame_stack", 1),
        n_rays=cfg.get("n_rays", 36),
        ray_range_px=cfg.get("ray_range_px", 64.0),
        arc_horizon=cfg.get("arc_horizon", 45),
    )
    env_config = CurveEnvConfig(
        engine_resolution=cfg["engine_resolution"],
        obs=obs_cfg,
        enabled_items=set(),
        max_episode_ticks=cfg.get("max_episode_ticks", 3600),
        reward=RewardConfig(**cfg.get("reward", {})),
    )
    base_planner = cfg.get("planner", {})
    # planner_mix: Liste von {weight, ...Overrides}. Leer = nur die Basiseinstellung.
    mix = cfg.get("planner_mix") or [{"weight": 1.0}]
    variants = []
    for entry in mix:
        entry = dict(entry)
        weight = float(entry.pop("weight", 1.0))
        variants.append((weight, PlannerConfig(**{**base_planner, **entry})))
    pcfg = variants[0][1]
    rng = random.Random(seed + 1000 * worker_id)
    env = CurveEnv(_episode_factory(obs_cfg, cfg.get("opponents", ["random"]), rng), config=env_config, seed=seed)
    planners = [(w, BeamPlanner(c, rng)) for w, c in variants]
    weights = [w for w, _ in planners]

    timer = PhaseTimer(out / "timing", run_label=f"collect_w{worker_id}")
    lengths, total_ticks = [], 0
    with timer.phase("collect", episodes=episodes, worker=worker_id):
        for ep in range(episodes):
            ep_seed = seed + 10_000 * worker_id + ep
            planner = rng.choices([p for _, p in planners], weights=weights, k=1)[0]
            with timer.section("episode"):
                data = collect_episode(env, planner, ep_seed)
            lengths.append(int(len(data["actions"])))
            total_ticks += lengths[-1]
            # komprimiert: die Frames sind grosse, sehr einfarbige uint8-Flaechen,
            # das packt zlib gut zusammen
            np.savez_compressed(shards / f"ep_w{worker_id}_{ep:05d}.npz", **data)

    index = {
        "worker": worker_id,
        "episodes": episodes,
        "total_ticks": total_ticks,
        "mean_episode_ticks": float(np.mean(lengths)) if lengths else 0.0,
        "obs": asdict(obs_cfg),
        "planner": asdict(pcfg),
        "planner_mix": [{"weight": w, **asdict(p.cfg)} for w, p in planners],
        "fatal_episodes": int(sum(1 for L in lengths if L < env_config.max_episode_ticks)),
        "created": time.time(),
    }
    (out / f"index_w{worker_id}.json").write_text(json.dumps(index, indent=2))
    print(f"[collect] worker {worker_id}: {episodes} Episoden, {total_ticks:,} Ticks, ø {index['mean_episode_ticks']:.0f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Expertentrajektorien mit dem Beam-Planer sammeln")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worker-id", type=int, default=0, help="beim parallelen Sammeln je Prozess eindeutig setzen")
    args = ap.parse_args()
    collect(args.config, args.out, args.episodes, seed=args.seed, worker_id=args.worker_id)


if __name__ == "__main__":
    main()
