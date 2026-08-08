"""Behavioral tests for CurveEngine: collision (border/self/other), scoring/kill
attribution, item effects (buff+revert, enemy debuff, r_swap's Phase 0 change,
b_clear), and the border-check's "unscaled hitbox" quirk preserved from script.js.
"""

import math

from ai.core.config import game_constants as gc
from ai.core.config.game_constants import GameConstants
from ai.core.env.engine import STRAIGHT, TURN_LEFT, CurveEngine, ItemOnScreen


def _engine(resolution=200):
    return CurveEngine(GameConstants(resolution))


def test_solo_border_death():
    # needs enough resolution that hitbox_size clears ~1px of rasterization rounding
    # noise (below ~S=200 straight-line movement can spuriously self-collide - see
    # ai/tests/test_engine.py's docstring note and the engine_resolution defaults)
    e = _engine(300)
    e.reset(["fred"], enabled_items=set(), seed=0)
    p = e.players["fred"]
    p.x, p.y, p.dir = e.c.engine_resolution / 2, e.c.engine_resolution / 2, 0.0

    for _ in range(2000):
        infos = e.step({"fred": STRAIGHT})
        if infos["fred"].just_died:
            assert infos["fred"].death_cause == "border"
            return
    raise AssertionError("player never died flying straight at the border")


def test_solo_episode_never_ends_from_alive_count_alone():
    # engine.is_episode_over() is a bare alive_count<=1 check - with only 1 player
    # that's true from tick 0, so curve_env.py must gate it on "opponents exist" (it
    # does - this test just pins the engine-level fact that callers must handle).
    e = _engine(200)
    e.reset(["fred"], enabled_items=set(), seed=0)
    assert e.alive_count() == 1
    assert e.is_episode_over() is True


def test_continuous_turn_causes_self_collision_not_border():
    e = _engine(200)
    e.reset(["fred"], enabled_items=set(), seed=0)
    p = e.players["fred"]
    p.x, p.y, p.dir = e.c.engine_resolution / 2, e.c.engine_resolution / 2, 0.0

    for _ in range(3000):
        infos = e.step({"fred": TURN_LEFT})
        if infos["fred"].just_died:
            assert infos["fred"].death_cause == "self"
            return
    raise AssertionError("player never self-collided while turning continuously in place")


def test_collision_with_other_players_trail_attributes_kill_and_score():
    e = _engine(200)
    e.reset(["fred", "greenlee"], enabled_items=set(), seed=0)
    a, b = e.players["fred"], e.players["greenlee"]
    a.x, a.y, a.dir = 100.0, 100.0, 0.0
    b.x, b.y, b.dir = 20.0, 20.0, math.pi  # out of A's way

    front_dist = e.c.hitbox_size * a.size
    target_x = round(a.x + front_dist * 1.02)  # slightly beyond where A's front sample lands post-move
    e.grid[round(a.y), target_x] = b.slot

    infos = e.step({"fred": STRAIGHT, "greenlee": STRAIGHT})
    assert infos["fred"].just_died
    assert infos["fred"].death_cause == "greenlee"
    assert b.kills == 1
    assert b.score == 1
    assert a.score == 0


def test_ghost_is_immune_to_own_trail_only():
    e = _engine(200)
    e.reset(["fred", "greenlee"], enabled_items=set(), seed=0)
    a, b = e.players["fred"], e.players["greenlee"]
    a.x, a.y, a.dir = 100.0, 100.0, 0.0
    a.ghost = 1
    b.x, b.y = 20.0, 20.0

    front_dist = e.c.hitbox_size * a.size
    target_x = round(a.x + front_dist * 1.02)
    e.grid[round(a.y), target_x] = a.slot  # A's own trail directly ahead

    infos = e.step({"fred": STRAIGHT, "greenlee": STRAIGHT})
    assert infos["fred"].alive  # ghost makes A immune to its OWN trail specifically

    a.ghost = 0
    e.grid[round(a.y), round(a.x + front_dist * 1.02)] = a.slot
    infos = e.step({"fred": STRAIGHT, "greenlee": STRAIGHT})
    assert infos["fred"].just_died
    assert infos["fred"].death_cause == "self"


def test_border_check_ignores_size_powerup():
    # script.js border check uses the *base* hitboxSize, not hitboxSize*size - so a
    # thickened (r_thick) player doesn't die earlier near the wall than a normal one.
    e = _engine(100)
    e.reset(["fred"], enabled_items=set(), seed=0)
    p = e.players["fred"]
    p.size = 3.0  # much thicker; border margin must NOT scale with this
    b = e.c.border_width + e.c.hitbox_size + e.field_inset
    p.x, p.y, p.dir = b - 0.5, e.c.engine_resolution / 2, math.pi  # just past the (unscaled) threshold, heading further out

    infos = e.step({"fred": STRAIGHT})
    assert infos["fred"].just_died and infos["fred"].death_cause == "border"


def test_g_fast_buffs_then_reverts_after_self_buff_ticks():
    e = _engine(200)
    e.reset(["fred"], enabled_items={"g_fast"}, seed=0)
    p = e.players["fred"]
    p.x, p.y, p.dir = 100.0, 100.0, 0.0
    front_dist = e.c.hitbox_size * p.size
    e.items_on_screen.append(ItemOnScreen("g_fast", p.x + front_dist, p.y))

    e.step({"fred": STRAIGHT})
    assert p.speed == 2.0
    assert p.items_collected == 1

    for _ in range(gc.SELF_BUFF_TICKS):
        e.step({"fred": STRAIGHT})
    assert abs(p.speed - 1.0) < 1e-9


def test_r_freeze_targets_all_others_regardless_of_alive_and_lasts_500ms():
    e = _engine(200)
    e.reset(["fred", "greenlee", "pinkney"], enabled_items={"r_freeze"}, seed=0)
    a, b, c_ = e.players["fred"], e.players["greenlee"], e.players["pinkney"]
    c_.alive = False  # even a dead slot gets the counter bump in the original (harmless)

    e._apply_item("r_freeze", "fred")
    assert b.freeze == 1
    assert c_.freeze == 1
    assert a.freeze == 0  # never applies to the picker

    for _ in range(gc.FREEZE_TICKS - 1):
        e._run_scheduled()
        e.tick += 1
    assert b.freeze == 1
    e.tick += 1
    e._run_scheduled()
    assert b.freeze == 0


def test_r_swap_exchanges_position_and_heading():
    # Phase 0 change: r_swap swaps heading too, not just position - each player
    # continues as if they'd physically become the other, in their own color.
    e = _engine(200)
    e.reset(["fred", "greenlee"], enabled_items={"r_swap"}, seed=0)
    a, b = e.players["fred"], e.players["greenlee"]
    a.x, a.y, a.dir = 10.0, 10.0, 0.1
    b.x, b.y, b.dir = 90.0, 90.0, 2.5

    e._apply_item("r_swap", "fred")

    assert (a.x, a.y, a.dir) == (90.0, 90.0, 2.5)
    assert (b.x, b.y, b.dir) == (10.0, 10.0, 0.1)


def test_b_clear_wipes_the_grid():
    e = _engine(50)
    e.reset(["fred"], enabled_items={"b_clear"}, seed=0)
    e.grid[10, 10] = 1
    e._apply_item("b_clear", "fred")
    assert e.grid.sum() == 0


def test_straight_line_reaches_border_from_any_heading():
    # Regression fuer den Insta-Selbstkollisions-Bug: PILs dicke Linie + binaeres
    # Owner-Grid toeteten Geradeausfahrt frueher fuer die meisten Richtungen an der
    # eigenen, gerade gezeichneten Spur (der Browser ueberlebt dank Canvas-
    # Antialiasing und der alpha==255-Pruefung). Mit der Frische-Grace-Period muss
    # jede Richtung sauber bis zur Wand kommen.
    e = _engine(256)
    for k in range(24):
        e.reset(["fred"], enabled_items=set(), seed=0)
        p = e.players["fred"]
        p.x = p.y = 128.0
        p.dir = k / 24 * 2 * math.pi
        cause = None
        for _ in range(600):
            info = e.step({"fred": STRAIGHT})["fred"]
            if not info.alive:
                cause = info.death_cause
                break
        assert cause == "border", f"Richtung {k}/24: starb an {cause!r} statt die Wand zu erreichen"
