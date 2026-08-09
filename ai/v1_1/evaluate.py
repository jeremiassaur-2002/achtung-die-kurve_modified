"""v1_1 gegen die Referenzen messen - in der echten Engine, nicht im Traum.

    python -m ai.v1_1.evaluate --run ai/output/v1_1/<lauf> --episodes 20

Drei Steuerungen auf identischen Startseeds:

  `dreamer`  der trainierte Agent
  `planner`  der Beam-Planer (der Experte, aus dem der Datensatz stammt)
  `rules`    der bestehende Regel-Bot aus v1_0

Identische Seeds sind kein Detail: die Startposition und -richtung streuen die
Ueberlebensdauer stark, und ohne gepaarte Vergleiche braucht man ein Vielfaches
an Episoden fuer dieselbe Aussagekraft.

**Woran der Erfolg haengt.** Den Planer zu schlagen ist die eigentliche Huerde
und keineswegs selbstverstaendlich: der Planer sieht das exakte Gitter, der
Dreamer nur ein 64x64-Bild plus Sensoren. Sein Vorteil liegt woanders - er kann
lernen, was Gegner als naechstes zeichnen, was der Planer prinzipiell nicht
kann (er friert das Gitter ein). Bleibt der Dreamer solo unter dem Planer, aber
mit Gegnern darueber, ist genau das passiert - und das ist ein Erfolg, kein
Widerspruch.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import yaml

from ai.core.config.game_constants import GameConstants
from ai.core.env.engine import CurveEngine
from ai.core.env.observation import ObsConfig
from ai.core.env.rules_bot import RuleBasedBot
from ai.v1_1.planner.beam import BeamPlanner, PlannerConfig


def survive_episode(decide, reset, seed: int, engine_resolution: int, cap: int, seats: list[str]) -> int:
    engine = CurveEngine(GameConstants(engine_resolution))
    engine.reset(seats, seed=seed)
    reset()
    rng = random.Random(seed)
    hero = seats[0]
    ticks = 0
    while engine.players[hero].alive and ticks < cap:
        actions = {hero: decide(engine, hero)}
        for other in seats[1:]:
            actions[other] = rng.choice([0, 1, 2])
        engine.step(actions)
        ticks += 1
    return ticks


def evaluate(run_dir: Path, config_path: Path, episodes: int = 20, cap: int = 3000, opponents: int = 0) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    obs_cfg = ObsConfig(
        obs_resolution=cfg["obs_resolution"],
        frame_stack=1,
        n_rays=cfg.get("n_rays", 36),
        ray_range_px=cfg.get("ray_range_px", 64.0),
        arc_horizon=cfg.get("arc_horizon", 45),
    )
    seats = ["fred", "greenlee", "pinkney", "bluebell"][: 1 + opponents]
    res = cfg["engine_resolution"]

    contenders: dict[str, tuple] = {}

    planner = BeamPlanner(PlannerConfig(**cfg.get("planner", {})))
    contenders["planner"] = (lambda e, n: planner.act(e, n), lambda: None)

    bot = RuleBasedBot("hard", random.Random(0))
    contenders["rules"] = (lambda e, n: bot.decide(e, n), lambda: None)

    policy = run_dir / "best" / "policy.pt"
    world_model = run_dir / "best" / "world_model.pt"
    if policy.exists() and world_model.exists():
        from ai.v1_1.agent import DreamerAgent

        agent = DreamerAgent.load(policy, world_model, obs_cfg)
        contenders["dreamer"] = (lambda e, n: agent.act(e, n), agent.reset)
    else:
        print(f"[eval] kein trainierter Agent unter {run_dir}/best - vergleiche nur die Referenzen")

    results: dict[str, dict] = {}
    for label, (decide, reset) in contenders.items():
        lengths = [survive_episode(decide, reset, seed, res, cap, seats) for seed in range(episodes)]
        results[label] = {
            "mean": statistics.mean(lengths),
            "median": statistics.median(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "lengths": lengths,
        }
        print(f"[eval] {label:>8}: ø {results[label]['mean']:7.1f} | Median {results[label]['median']:7.1f} Ticks")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="v1_1-Agent gegen Planer und Regel-Bot")
    ap.add_argument("--run", required=True, help="Laufverzeichnis, z.B. ai/output/v1_1/<lauf>")
    ap.add_argument("--config", default=None, help="Default: <run>/config_used.yaml")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--opponents", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run)
    config = Path(args.config) if args.config else run_dir / "config_used.yaml"
    results = evaluate(run_dir, config, args.episodes, args.cap, args.opponents)
    out = run_dir / "reports" / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[eval] -> {out}")


if __name__ == "__main__":
    main()
