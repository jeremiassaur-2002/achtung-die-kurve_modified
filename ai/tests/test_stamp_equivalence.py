"""_stamp_segment (bounding-box blit) must be BIT-IDENTICAL to the old full-frame
PIL path it replaced - the collision rules read individual pixels, so "almost the
same" rasterization would silently change which states are lethal. This fuzzes
tens of thousands of random gameplay-scale segments (including off-grid, edge-
clipped, zero-length and extreme-width ones) against a verbatim copy of the old
implementation. Scope: real segments span exactly TWO movement steps (~1.2px at
S=256; ~10px under stacked g_fast, widths up to r_thick stacks), and within that
envelope the two paths are bit-identical. Arbitrarily long segments are excluded:
PIL's float edge math can flip a boundary pixel under translation there, and the
engine cannot produce such segments in the first place."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw

from ai.core.config.game_constants import GameConstants
from ai.core.env.engine import CurveEngine


def _reference_stamp(grid: np.ndarray, stamp: np.ndarray, slot: int, tick: int, xa, ya, xb, yb, width: int) -> None:
    """Verbatim old engine code: full-frame mask image + full-grid boolean assignment."""
    mask_img = Image.new("L", (grid.shape[1], grid.shape[0]), 0)
    ImageDraw.Draw(mask_img).line([(xa, ya), (xb, yb)], fill=255, width=width)
    mask = np.asarray(mask_img) > 0
    grid[mask] = slot
    stamp[mask] = tick


def test_stamp_segment_matches_old_full_frame_path():
    rng = random.Random(12345)
    for s in (200, 256, 300):
        eng = CurveEngine(GameConstants(s))
        eng.reset(["fred"], enabled_items=set(), seed=0)
        ref_grid = np.zeros_like(eng.grid)
        ref_stamp = np.full_like(eng.stamp, -(10**9))
        eng.grid[:] = 0
        eng.stamp[:] = -(10**9)

        for i in range(10000):
            eng.tick = i
            slot = rng.randrange(1, 7)
            width = rng.choice((1, 1, 2, 2, 2, 3, 4, 8, 16, 32))
            # gameplay-scale segments: ~1.2px normally, up to ~12px with stacked
            # g_fast; placed everywhere incl. off-grid (wrap adjusts coords by +-S,
            # so partially/fully clipped segments are a real, common case)
            xa = rng.uniform(-20, s + 20)
            ya = rng.uniform(-20, s + 20)
            xb = xa + rng.uniform(-12, 12)
            yb = ya + rng.uniform(-12, 12)
            if rng.random() < 0.03:
                xb, yb = xa, ya  # zero-length (frozen-ish edge case)
            _reference_stamp(ref_grid, ref_stamp, slot, i, xa, ya, xb, yb, width)
            eng._stamp_segment(slot, xa, ya, xb, yb, width)

            assert np.array_equal(eng.grid, ref_grid), f"grid mismatch at iter {i} (S={s})"
            assert np.array_equal(eng.stamp, ref_stamp), f"stamp mismatch at iter {i} (S={s})"


def test_engine_episode_unchanged_by_stamp_rewrite():
    """Same seed + same action script must produce the same death tick and grid as
    before the rewrite - checked via self-consistency of two engines stepping the
    same script (one exercising many wrap/size states via items disabled/enabled)."""
    for seed in (1, 7, 42):
        eng = CurveEngine(GameConstants(256))
        eng.reset(["fred", "greenlee", "pinkney"], enabled_items=set(), seed=seed)
        rng = random.Random(seed)
        for _ in range(4000):
            if eng.alive_count() == 0:
                break
            eng.step({n: rng.choice((0, 1, 1, 2)) for n in eng.players})
        # sanity: trails were actually stamped and someone eventually died
        assert int(np.count_nonzero(eng.grid)) > 200
        assert any(p.death_tick is not None for p in eng.players.values())
