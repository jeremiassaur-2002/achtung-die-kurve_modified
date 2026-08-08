"""Automatic evaluation: win rate, survival time, placement, kills, item usage and
collisions for a checkpoint against random bots, every rule-based difficulty, and
(if a League is given) older checkpoints - the exact axes the brief asks for.

CLI:
    python -m ai.core.evaluation.evaluate --checkpoint path/to/model.zip
    python -m ai.core.evaluation.evaluate --checkpoint path/to/model.zip --league ai/league --matches 30
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass, field

from ai.core.config.game_constants import GameConstants
from ai.core.env.observation import ObsConfig
from ai.core.env.opponents import FrozenPolicyController, OpponentSpec, build_controller
from ai.core.evaluation.arena import run_match

CURRENT_ID = "current"
RULE_BASED_DIFFICULTIES = ("easy", "medium", "hard", "hunter")


@dataclass
class EvalSummary:
    opponent_label: str
    matches: int
    win_rate: float
    avg_placement: float  # 1.0 = always first
    avg_survival_ticks: float
    avg_kills: float
    avg_items_collected: float
    death_causes: dict[str, int] = field(default_factory=dict)


def evaluate_against(
    model_path: str,
    opponent_specs: list[OpponentSpec],
    obs_cfg: ObsConfig | None = None,
    constants: GameConstants | None = None,
    n_matches: int = 20,
    enabled_items: set[str] | None = None,
    max_ticks: int = 3600,
    seed: int = 0,
    label: str | None = None,
) -> EvalSummary:
    obs_cfg = obs_cfg or ObsConfig()
    constants = constants or GameConstants(256)
    rng = random.Random(seed)

    wins = 0
    placements_sum = 0
    survival_sum = 0
    kills_sum = 0
    items_sum = 0
    death_causes: Counter[str] = Counter()

    for i in range(n_matches):
        current_ctrl = FrozenPolicyController(model_path, obs_cfg, deterministic=True)
        participants = {CURRENT_ID: current_ctrl}
        for j, spec in enumerate(opponent_specs):
            participants[f"opp{j}"] = build_controller(spec, obs_cfg, rng)

        result = run_match(participants, constants, enabled_items, max_ticks=max_ticks, seed=seed + i)
        placement = result.placements.index(CURRENT_ID) + 1  # 1-based, 1 = best
        placements_sum += placement
        wins += 1 if placement == 1 else 0
        survival_sum += result.survival_ticks[CURRENT_ID]
        kills_sum += result.kills[CURRENT_ID]
        items_sum += result.items_collected[CURRENT_ID]
        cause = result.death_cause[CURRENT_ID]
        death_causes[cause or "survived"] += 1

    n = max(1, n_matches)
    return EvalSummary(
        opponent_label=label or _default_label(opponent_specs),
        matches=n_matches,
        win_rate=wins / n,
        avg_placement=placements_sum / n,
        avg_survival_ticks=survival_sum / n,
        avg_kills=kills_sum / n,
        avg_items_collected=items_sum / n,
        death_causes=dict(death_causes),
    )


def _default_label(specs: list[OpponentSpec]) -> str:
    kinds = [s.difficulty if s.kind == "rule_based" else s.kind for s in specs]
    return "+".join(kinds)


def evaluate_model(
    model_path: str,
    n_matches: int = 20,
    obs_cfg: ObsConfig | None = None,
    constants: GameConstants | None = None,
    league=None,
    league_samples: int = 3,
    seed: int = 0,
) -> dict[str, EvalSummary]:
    """The standard battery: 1 random opponent, each rule-based difficulty, and
    (if a League is supplied) a handful of historical checkpoints + the best one."""
    obs_cfg = obs_cfg or ObsConfig()
    constants = constants or GameConstants(256)
    summaries: dict[str, EvalSummary] = {}

    summaries["random"] = evaluate_against(
        model_path, [OpponentSpec(kind="random")], obs_cfg, constants, n_matches, seed=seed, label="random"
    )
    for diff in RULE_BASED_DIFFICULTIES:
        summaries[f"rule_based_{diff}"] = evaluate_against(
            model_path, [OpponentSpec(kind="rule_based", difficulty=diff)], obs_cfg, constants, n_matches, seed=seed, label=f"rule_based:{diff}"
        )

    if league is not None:
        rng = random.Random(seed)
        entries = sorted(league.entries.values(), key=lambda e: e.step)
        sample = entries[-league_samples:] if len(entries) > league_samples else entries
        for entry in sample:
            summaries[entry.name] = evaluate_against(
                model_path, [OpponentSpec(kind="frozen", model_path=entry.path)], obs_cfg, constants, n_matches, seed=seed, label=entry.name
            )
        best = league.best()
        if best is not None and best.name not in summaries:
            summaries[best.name] = evaluate_against(
                model_path, [OpponentSpec(kind="frozen", model_path=best.path)], obs_cfg, constants, n_matches, seed=seed, label=best.name
            )

    return summaries


def _print_summary(summaries: dict[str, EvalSummary]) -> None:
    header = f"{'opponent':<22}{'matches':>8}{'win_rate':>10}{'avg_place':>10}{'avg_surv':>10}{'avg_kills':>10}{'avg_items':>10}"
    print(header)
    print("-" * len(header))
    for key, s in summaries.items():
        print(f"{key:<22}{s.matches:>8}{s.win_rate:>10.2f}{s.avg_placement:>10.2f}{s.avg_survival_ticks:>10.1f}{s.avg_kills:>10.2f}{s.avg_items_collected:>10.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint against random/rule-based/league opponents.")
    parser.add_argument("--checkpoint", required=True, help="path to an SB3 .zip checkpoint")
    parser.add_argument("--league", default=None, help="path to a league directory (optional)")
    parser.add_argument("--matches", type=int, default=20)
    parser.add_argument("--engine-resolution", type=int, default=256)
    parser.add_argument("--obs-resolution", type=int, default=96)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    league = None
    if args.league:
        from ai.v1_0.training.league import League

        league = League(args.league)

    obs_cfg = ObsConfig(obs_resolution=args.obs_resolution, frame_stack=args.frame_stack)
    constants = GameConstants(args.engine_resolution)
    summaries = evaluate_model(args.checkpoint, n_matches=args.matches, obs_cfg=obs_cfg, constants=constants, league=league, seed=args.seed)
    _print_summary(summaries)


if __name__ == "__main__":
    main()
