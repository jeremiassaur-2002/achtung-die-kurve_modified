"""Checks every derived ratio against the literal script.js formulas, so a future
edit to game_constants.py can't silently drift from the source of truth."""

from ai.config import game_constants as gc
from ai.config.game_constants import GameConstants


def test_w100th_ratio_matches_4_3_aspect():
    # script.js: w100th = w / 100, w = (4/3) * h  =>  w100th = h / 75
    assert gc.W100TH_RATIO == 1.0 / 75.0


def test_ratios_at_S_75_give_whole_numbers():
    # at S=75, w100th == 1 exactly - makes the script.js constants read off directly
    c = GameConstants(75)
    assert abs(c.move_speed - 0.18) < 1e-9
    assert abs(c.player_size - 0.7) < 1e-9
    assert abs(c.hitbox_size - 0.7 / 1.8) < 1e-9
    assert abs(c.border_width - 0.5) < 1e-9
    assert abs(c.icon_size - 2.0) < 1e-9


def test_ratios_scale_linearly_with_resolution():
    c1 = GameConstants(256)
    c2 = GameConstants(512)
    for attr in ("move_speed", "player_size", "hitbox_size", "border_width", "icon_size", "start_inset"):
        assert abs(getattr(c2, attr) - 2 * getattr(c1, attr)) < 1e-9, attr


def test_turn_speed_is_not_scaled_by_resolution():
    # script.js: turnSpeed = 0.06 is a plain constant, independent of arena size
    assert gc.TURN_SPEED == 0.06


def test_powerup_timers_converted_from_ms_at_60hz():
    assert gc.TICK_RATE == 60
    assert gc.SELF_BUFF_TICKS == 480  # 8000ms
    assert gc.ENEMY_DEBUFF_TICKS == 300  # 5000ms
    assert gc.FREEZE_TICKS == 30  # Phase 0 change: 500ms (was 1000ms/60 ticks)
    assert gc.B_MORE_DELAYS_TICKS == (6, 12, 18)  # 100/200/300ms


def test_robot_item_removed():
    assert "g_robot" not in gc.ALL_ITEMS
    assert "r_robot" not in gc.ALL_ITEMS
    assert len(gc.ALL_ITEMS) == 20  # 22 originally, minus g_robot/r_robot


def test_player_roster_matches_script_js():
    assert gc.PLAYER_NAMES == ("fred", "greenlee", "pinkney", "bluebell", "willem", "greydon")
    assert gc.MAX_PLAYERS == 6
    assert gc.PLAYER_COLORS["fred"] == (255, 0, 0)
    assert gc.PLAYER_COLORS["greenlee"] == (0, 255, 0)


def test_scoring_rules():
    assert gc.POINT_GOAL_PER_OPPONENT == 10
    assert gc.WIN_MARGIN == 2
