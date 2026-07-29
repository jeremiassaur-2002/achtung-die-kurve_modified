"""Game constants extracted from the original "Achtung, die Kurve!" JS source
(script.js / style.css / powerupmenu.js), after the Phase 0 balance edits made
alongside this training system:

  - the "Roboter" item (g_robot / r_robot) was removed entirely
  - r_freeze duration is 500ms (was 1000ms)
  - r_swap now swaps position AND heading (was position-only)

Every ratio/timer below is reproduced from the JS source, not guessed. The JS game
has no fixed pixel resolution -- every gameplay constant is derived at runtime from
`w100th = w/100` where `w` is the *full* canvas width (arena + side panel) and the
arena itself is the square `h x h` with `w = 4/3 * h` (`#game { aspect-ratio: 4/3 }`
in style.css). That makes every constant an exact ratio of the arena size `h`, which
this module calls `S` (the chosen simulation/engine resolution).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tick rate
# ---------------------------------------------------------------------------

# script.js draw(): the simulation is fixed at 60 steps/second via a time
# accumulator, decoupled from the browser's actual display refresh rate.
TICK_RATE = 60

# ---------------------------------------------------------------------------
# Ratios relative to arena size S (script.js newSize())
# ---------------------------------------------------------------------------

# w100th = w / 100, w = (4/3) * S  =>  w100th = S / 75
W100TH_RATIO = 1.0 / 75.0

MOVE_SPEED_RATIO = W100TH_RATIO * 0.18       # moveSpeed = w100th * 0.18
PLAYER_SIZE_RATIO = W100TH_RATIO * 0.7        # playerSize = w100th * 0.7
HITBOX_SIZE_RATIO = PLAYER_SIZE_RATIO / 1.8   # hitboxSize = playerSize / 1.8
BORDER_WIDTH_RATIO = W100TH_RATIO / 2.0       # borderWidth = w100th / 2
ICON_SIZE_RATIO = W100TH_RATIO * 2.0          # iconSize = w100th * 2

# turnSpeed is a plain constant in script.js, NOT scaled by arena size.
TURN_SPEED = 0.06  # rad/tick while turning (continuous steering)
SIZE_TURN_EXPONENT = 0.3  # dir += turnSpeed / size^0.3  (g_thin turns faster, r_thick slower)

# g_sine/r_sine: turn magnitude scaled by 2**sin(...) over this many ticks, sign untouched.
SINE_TURN_DURATION_TICKS = 240

# script.js pxFront2/pyFront2: an extra near-field collision sample ~1 raw unit ahead.
FRONT2_SAMPLE_DIST = 1.0
LEFT_RIGHT_ANGLE_DEG = 55.0  # left/right hitbox sample points at +-55 degrees off heading

# ---------------------------------------------------------------------------
# Bridges (random gaps in your own trail) and item spawning
# ---------------------------------------------------------------------------

BRIDGE_PROB = 0.005          # per-tick chance to open a bridge (script.js bridgeProb)
BRIDGE_SIZE_TICKS = 10       # base bridge duration; actual = BRIDGE_SIZE_TICKS/speed*size
POWERUP_SPAWN_PROB = 0.005   # per-tick chance to spawn an item, when items are enabled
POWERUP_MAX_ON_SCREEN = 30
START_INSET_BORDER_MULT = 10  # random spawn inset = borderWidth * 10, to avoid instant death

# ---------------------------------------------------------------------------
# Powerup effect durations, converted ms -> ticks @ TICK_RATE
# ---------------------------------------------------------------------------


def _ms_to_ticks(ms: float) -> int:
    return round(ms * TICK_RATE / 1000.0)


SELF_BUFF_TICKS = _ms_to_ticks(8000)     # 480 - g_* buffs, and b_sides/b_shrink/b_grow reverts
ENEMY_DEBUFF_TICKS = _ms_to_ticks(5000)  # 300 - r_* debuffs, except r_freeze
FREEZE_TICKS = _ms_to_ticks(500)         # 30  - r_freeze (Phase 0: shortened from 1000ms)
B_MORE_DELAYS_TICKS = tuple(_ms_to_ticks(ms) for ms in (100, 200, 300))  # (6, 12, 18)

# ---------------------------------------------------------------------------
# Scoring / win condition (script.js checkGameState/drawGameUI/givePoints)
# ---------------------------------------------------------------------------

POINT_GOAL_PER_OPPONENT = 10  # pointGoal = (numPlayers - 1) * 10
WIN_MARGIN = 2                # leader must be >= 2 points ahead of 2nd place to win the match

# ---------------------------------------------------------------------------
# Players and colors (script.js `players` object; style.css :root palette)
# ---------------------------------------------------------------------------

PLAYER_NAMES = ("fred", "greenlee", "pinkney", "bluebell", "willem", "greydon")
MAX_PLAYERS = len(PLAYER_NAMES)  # 6 total seats -> 1 hero + up to 5 opponents

PLAYER_COLORS: dict[str, tuple[int, int, int]] = {
    "fred": (255, 0, 0),
    "greenlee": (0, 255, 0),
    "pinkney": (255, 0, 255),
    "bluebell": (0, 255, 255),
    "willem": (255, 127, 0),
    "greydon": (200, 200, 200),
}

HEAD_COLOR_NORMAL = (255, 255, 0)   # yellow head dot
HEAD_COLOR_REVERSED = (0, 0, 255)   # blue head dot while r_reverse is active
BORDER_COLOR = (255, 255, 0)
BACKGROUND_COLOR = (0, 0, 0)
ITEM_SELF_COLOR = (0, 255, 0)     # g_* badge color family (green)
ITEM_ENEMY_COLOR = (255, 0, 0)    # r_* badge color family (red)
ITEM_GLOBAL_COLOR = (0, 0, 255)   # b_*/o_* badge color family (blue)

# ---------------------------------------------------------------------------
# Items (achtung.powerups in script.js, after Phase 0 removed g_robot/r_robot)
# ---------------------------------------------------------------------------

SELF_ITEMS = ("g_slow", "g_fast", "g_thin", "g_side", "g_invisible", "g_sine", "g_ghost")
ENEMY_ITEMS = ("r_slow", "r_fast", "r_thick", "r_reverse", "r_sine", "r_swap", "r_freeze")
GLOBAL_ITEMS = ("b_clear", "b_more", "b_sides", "b_shrink", "b_grow")
RANDOM_ITEM = "o_random"
ALL_ITEMS = SELF_ITEMS + ENEMY_ITEMS + GLOBAL_ITEMS + (RANDOM_ITEM,)


@dataclass(frozen=True)
class GameConstants:
    """All physical constants instantiated at a concrete arena resolution `S`.

    `S` plays the exact role of `h` (the square arena side length in pixels) in
    script.js's `newSize()`. Collision/physics fidelity depends only on this
    number, not on how big the CNN's observation image is (see env/renderer.py).

    Keep `S` >= ~200: below that, `hitbox_size` drops under ~1px and the engine's
    integer-pixel rasterization (no canvas-style anti-aliasing) makes straight-
    line movement spuriously clip its own just-drawn trail. The default (256)
    clears this with a small margin; this was found empirically, not assumed.
    """

    engine_resolution: int

    @property
    def move_speed(self) -> float:
        return self.engine_resolution * MOVE_SPEED_RATIO

    @property
    def player_size(self) -> float:
        return self.engine_resolution * PLAYER_SIZE_RATIO

    @property
    def hitbox_size(self) -> float:
        return self.engine_resolution * HITBOX_SIZE_RATIO

    @property
    def border_width(self) -> float:
        return self.engine_resolution * BORDER_WIDTH_RATIO

    @property
    def icon_size(self) -> float:
        return self.engine_resolution * ICON_SIZE_RATIO

    @property
    def start_inset(self) -> float:
        return self.border_width * START_INSET_BORDER_MULT

    @property
    def shrink_step(self) -> float:
        return self.engine_resolution * 0.08  # b_shrink: fieldInset += S*0.08, capped at S*0.3

    @property
    def shrink_cap(self) -> float:
        return self.engine_resolution * 0.3

    @property
    def grow_step(self) -> float:
        return self.engine_resolution * 0.05  # b_grow: fieldInset -= S*0.05

    @property
    def grow_cap(self) -> float:
        return -self.hitbox_size * 0.8

    def __post_init__(self) -> None:
        if self.engine_resolution <= 0:
            raise ValueError("engine_resolution must be positive")
