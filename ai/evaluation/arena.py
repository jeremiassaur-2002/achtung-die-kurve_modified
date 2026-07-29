"""Runs one full headless match (one round: play until <=1 player is left
standing, or a tick cap) between arbitrary Controllers. Used by evaluate.py for
win-rate/survival/kills/item-usage metrics, and by league Elo updates - both just
need a placement order plus per-participant stats out of a match.

Participants are keyed by a stable *identity* string (e.g. "current_model",
"checkpoint_50", "rule_based:hunter", "random"), not a game seat/color name -
arena.py randomly assigns seats internally and translates back, so identities
stay meaningful across matches even though seats/colors are randomized each time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ai.config import game_constants as gc
from ai.config.game_constants import GameConstants
from ai.env.engine import CurveEngine
from ai.env.opponents import Controller


@dataclass
class MatchResult:
    placements: list[str]  # identities, best (index 0) to worst
    survival_ticks: dict[str, int] = field(default_factory=dict)
    kills: dict[str, int] = field(default_factory=dict)
    items_collected: dict[str, int] = field(default_factory=dict)
    death_cause: dict[str, str | None] = field(default_factory=dict)


def run_match(
    participants: dict[str, Controller],
    constants: GameConstants | None = None,
    enabled_items: set[str] | None = None,
    max_ticks: int = 3600,
    seed: int | None = None,
) -> MatchResult:
    identities = list(participants.keys())
    if not (1 <= len(identities) <= gc.MAX_PLAYERS):
        raise ValueError(f"need 1..{gc.MAX_PLAYERS} participants, got {len(identities)}")

    constants = constants or GameConstants(256)
    rng = random.Random(seed)
    seat_names = list(gc.PLAYER_NAMES)
    rng.shuffle(seat_names)
    seat_for_identity = dict(zip(identities, seat_names[: len(identities)]))

    engine = CurveEngine(constants)
    engine.reset(list(seat_for_identity.values()), enabled_items=enabled_items, seed=seed)
    for identity, ctrl in participants.items():
        ctrl.reset(seat_for_identity[identity])

    death_order_seats: list[str] = []
    last_tick = 0
    for tick in range(max_ticks):
        last_tick = tick
        actions = {}
        for identity, ctrl in participants.items():
            seat = seat_for_identity[identity]
            if engine.players[seat].alive:
                actions[seat] = ctrl.act(engine, seat, None)
        infos = engine.step(actions)
        for seat, info in infos.items():
            if info.just_died:
                death_order_seats.append(seat)
        if engine.alive_count() <= 1:
            break

    survivor_seats = [s for s in seat_for_identity.values() if engine.players[s].alive]
    placement_seats = survivor_seats + list(reversed(death_order_seats))
    identity_for_seat = {seat: identity for identity, seat in seat_for_identity.items()}
    placements = [identity_for_seat[s] for s in placement_seats]

    return MatchResult(
        placements=placements,
        survival_ticks={identity: (engine.players[seat].death_tick or (last_tick + 1)) for identity, seat in seat_for_identity.items()},
        kills={identity: engine.players[seat].kills for identity, seat in seat_for_identity.items()},
        items_collected={identity: engine.players[seat].items_collected for identity, seat in seat_for_identity.items()},
        death_cause={identity: engine.players[seat].death_cause for identity, seat in seat_for_identity.items()},
    )
