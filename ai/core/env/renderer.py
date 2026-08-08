"""Rasterizes a CurveEngine's state into an RGB frame.

The frame is shared by every viewer (the hero and any frozen-policy opponents) -
colors are the game's real, fixed per-slot palette (see ai/core/config/game_constants.py),
not remapped per-perspective. A policy that can occupy any of the 6 seats is told
*which* color is "its own" via the vector observation built in curve_env.py, exactly
like a human learns to track their own colored trail on screen.

Composition order mirrors the original's canvas stacking (trails below, then items,
then the border, then player head dots on top - script.js/style.css z-index order):
background -> trails (recolored owner-id grid) -> items -> border -> head dots.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ai.core.config import game_constants as gc
from ai.core.env.engine import CurveEngine

_RESAMPLE = Image.Resampling.BOX  # good downsampling behavior for thin 1px trail lines


def _owner_palette(engine: CurveEngine) -> np.ndarray:
    """(max_slot+1, 3) uint8 lookup table: index 0 = background, index = player.slot."""
    palette = np.zeros((gc.MAX_PLAYERS + 1, 3), dtype=np.uint8)
    palette[0] = gc.BACKGROUND_COLOR
    for p in engine.players.values():
        palette[p.slot] = gc.PLAYER_COLORS[p.name]
    return palette


def render_frame(engine: CurveEngine, obs_resolution: int) -> np.ndarray:
    """Returns an (obs_resolution, obs_resolution, 3) uint8 RGB array (HWC)."""
    palette = _owner_palette(engine)
    rgb = palette[engine.grid]  # (S, S, 3) uint8, vectorized recolor of the owner-id grid

    img = Image.fromarray(rgb)  # (H, W, 3) uint8 -> PIL infers "RGB" without a mode= arg
    draw = ImageDraw.Draw(img)

    c = engine.c
    s = c.engine_resolution

    # items - small colored badges by family; o_random drawn white (script.js's tri-color
    # wheel badge doesn't map cleanly onto a single family color, so it gets its own marker)
    item_r = max(1, round(c.icon_size))
    for item in engine.items_on_screen:
        if item.kind in gc.SELF_ITEMS:
            color = gc.ITEM_SELF_COLOR
        elif item.kind in gc.ENEMY_ITEMS:
            color = gc.ITEM_ENEMY_COLOR
        elif item.kind == gc.RANDOM_ITEM:
            color = (255, 255, 255)
        else:
            color = gc.ITEM_GLOBAL_COLOR
        draw.ellipse([item.x - item_r, item.y - item_r, item.x + item_r, item.y + item_r], fill=color)

    # border, inset by field_inset (b_shrink/b_grow) - matches script.js's strokeRect
    inset = engine.field_inset
    bw = max(1, round(c.border_width))
    x0, y0 = bw / 2 + inset, bw / 2 + inset
    x1, y1 = s - bw / 2 - inset, s - bw / 2 - inset
    draw.rectangle([x0, y0, x1, y1], outline=gc.BORDER_COLOR, width=bw)

    # head dots - always yellow/blue regardless of player identity (only trails are
    # per-player colored in the original), drawn last so they sit on top of everything
    head_r = c.player_size / 2
    for p in engine.players.values():
        if not p.alive:
            continue
        r = head_r * p.size
        draw.ellipse([p.x - r, p.y - r, p.x + r, p.y + r], fill=p.head_color)

    if s != obs_resolution:
        img = img.resize((obs_resolution, obs_resolution), _RESAMPLE)

    return np.asarray(img, dtype=np.uint8)


def to_chw(frame_hwc: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, H, W) uint8, the channel-first layout SB3's CNN policies expect."""
    return np.transpose(frame_hwc, (2, 0, 1))
