"""Rotating, resumable checkpoints + headless episode-video recording.

Deliberately torch/SB3-free: everything here is plain filesystem + numpy/PIL
work, so it stays unit-testable without the RL stack (see ai/tests). The thin
SB3 callbacks that use these helpers live in ai/training/drive_callbacks.py.

Rotation contract (important for Colab + Google Drive): a new checkpoint is
FULLY written (temp name, then rename) before any older one is deleted. If the
runtime dies mid-write, the previous checkpoint is still intact - you can never
end up with zero usable checkpoints. With the notebook's Drive-mounted
--run-root, these files land directly on Google Drive.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

_CKPT_RE = re.compile(r"^ckpt_(\d+)_steps\.zip$")


def checkpoint_path(ckpt_dir: str | Path, steps: int) -> Path:
    return Path(ckpt_dir) / f"ckpt_{steps}_steps.zip"


def checkpoint_steps(path: str | Path) -> int:
    m = _CKPT_RE.match(Path(path).name)
    return int(m.group(1)) if m else -1


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    """Newest (highest-step) valid checkpoint in `ckpt_dir`, or None."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    best, best_steps = None, -1
    for f in ckpt_dir.iterdir():
        steps = checkpoint_steps(f)
        if steps > best_steps:
            best, best_steps = f, steps
    return best


def save_rotating_checkpoint(model, ckpt_dir: str | Path, steps: int, keep: int = 1) -> Path:
    """Save `model` as ckpt_<steps>_steps.zip, then prune older checkpoints so at
    most `keep` remain. Write order (temp -> rename -> prune) guarantees the old
    checkpoint only disappears once the new one is complete."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = checkpoint_path(ckpt_dir, steps)
    tmp = final.with_name(final.name + ".part")
    model.save(str(tmp))  # SB3 keeps the exact name (it only appends ".zip" when there is no suffix at all)
    os.replace(tmp, final)

    kept = sorted((f for f in ckpt_dir.iterdir() if checkpoint_steps(f) >= 0), key=checkpoint_steps, reverse=True)
    for old in kept[max(1, keep):]:
        old.unlink(missing_ok=True)
    for stray in ckpt_dir.glob("*.part"):  # leftovers from an earlier crash mid-write
        if stray != tmp:
            stray.unlink(missing_ok=True)
    return final


class HiResEpisodeRenderer:
    """Draws the running episode VECTORIALLY in parallel on a high-resolution canvas.

    Why: the engine has no high-res image of the world - trails exist ONLY as the
    S x S owner-id grid (256px), so every video was a nearest-neighbor upscale of
    that raster: chunky 2x2/3x3 blocks, sub-pixel head dots. Physics stays untouched
    at S (collision fidelity is engine_resolution's job); this class merely REDRAWS
    what the engine just did - one line segment per player per tick, items, border,
    head dots - at `render_scale` x S, internally 2x supersampled and box-downsampled
    for smooth edges. Presentation only: nothing here is read back into the game.

    Faithfulness notes (deliberate, all cosmetic):
    - segments are drawn prev->current with round caps (the engine draws
      prevprev->current with butt caps; round caps close the joints the same way
      the doubled segment does, and the head dot covers the tip in both cases)
    - a b_clear pickup is detected via the owner-grid's nonzero count dropping
      (the grid never shrinks otherwise) and clears the trail canvas
    - wrap (sides): the stored previous point is shifted by +-S toward the new
      position, so only the entering stub is drawn - exactly like the engine
    - a border death happens BEFORE the tick's segment is stamped, a trail death
      AFTER it - mirrored here via death_tick/death_cause, so the final frame
      shows precisely the ink that exists in the engine grid
    """

    def __init__(self, engine, render_scale: int = 3, supersample: int = 2):
        from PIL import Image

        self.s = engine.c.engine_resolution
        self.k = max(1, int(render_scale))
        ss = max(1, int(supersample))
        while ss > 1 and self.s * self.k * ss > 2048:  # cap the internal canvas at 2048^2 (RAM)
            ss -= 1
        self.ss = ss
        self.K = self.k * self.ss
        from ai.config import game_constants as gc

        self._trails = Image.new("RGB", (self.s * self.K, self.s * self.K), gc.BACKGROUND_COLOR)
        self._prev: dict[str, tuple[float, float]] = {n: (p.x, p.y) for n, p in engine.players.items()}
        self._last_nonzero = int(np.count_nonzero(engine.grid))

    def update(self, engine) -> None:
        """Call once after every engine tick (i.e. after env.step)."""
        from PIL import ImageDraw

        nz = int(np.count_nonzero(engine.grid))
        if nz < self._last_nonzero:  # only b_clear ever removes pixels
            from ai.config import game_constants as gc

            self._trails.paste(gc.BACKGROUND_COLOR, (0, 0, self._trails.width, self._trails.height))
        self._last_nonzero = nz

        draw = ImageDraw.Draw(self._trails)
        s, K = self.s, self.K
        for name, p in engine.players.items():
            stepped = p.alive or (p.death_tick == engine.tick and p.death_cause != "border")
            if not stepped:
                continue
            px, py = self._prev.get(name, (p.x, p.y))
            # wrap: shift the stored point onto the new side, per axis (engine does
            # the same to prevPos, so only the entering stub gets ink)
            if p.x - px > s / 2:
                px += s
            elif p.x - px < -s / 2:
                px -= s
            if p.y - py > s / 2:
                py += s
            elif p.y - py < -s / 2:
                py -= s
            if not p.bridge and p.invisible == 0:
                w = max(1, round(engine.c.player_size * p.size * K))
                color = p.color
                draw.line([(px * K, py * K), (p.x * K, p.y * K)], fill=color, width=w)
                r = w / 2
                for cx, cy in ((px * K, py * K), (p.x * K, p.y * K)):  # round caps
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
            self._prev[name] = (p.x, p.y)

    def frame(self, engine) -> np.ndarray:
        """(S*render_scale, S*render_scale, 3) uint8 frame: trails + items + border
        + head dots, composed in the same z-order as ai/env/renderer.py."""
        from PIL import Image, ImageDraw

        from ai.config import game_constants as gc

        img = self._trails.copy()
        draw = ImageDraw.Draw(img)
        c = engine.c
        s, K = self.s, self.K

        item_r = c.icon_size * K
        for item in engine.items_on_screen:
            fill = _item_color(item.kind)
            draw.ellipse([item.x * K - item_r, item.y * K - item_r, item.x * K + item_r, item.y * K + item_r], fill=fill)

        inset = engine.field_inset * K
        bw = max(1, round(c.border_width * K))
        x0 = bw / 2 + inset
        x1 = s * K - bw / 2 - inset
        draw.rectangle([x0, x0, x1, x1], outline=gc.BORDER_COLOR, width=bw)

        head_r = (c.player_size / 2) * K
        for p in engine.players.values():
            if not p.alive:
                continue
            r = head_r * p.size
            draw.ellipse([p.x * K - r, p.y * K - r, p.x * K + r, p.y * K + r], fill=p.head_color)

        if self.ss > 1:
            img = img.resize((s * self.k, s * self.k), Image.Resampling.BOX)
        return np.asarray(img, dtype=np.uint8)


def _item_color(kind: str) -> tuple[int, int, int]:
    from ai.config import game_constants as gc

    if kind in gc.SELF_ITEMS:
        return gc.ITEM_SELF_COLOR
    if kind in gc.ENEMY_ITEMS:
        return gc.ITEM_ENEMY_COLOR
    if kind == gc.RANDOM_ITEM:
        return (255, 255, 255)
    return gc.ITEM_GLOBAL_COLOR


def record_episode_video(
    env,
    predict_fn,
    out_path: str | Path,
    max_ticks: int = 1800,
    fps: int = 60,
    scale: int = 2,
    render_scale: int | None = 3,
):
    """Plays ONE episode in `env` (a CurveEnv) with `predict_fn(obs) -> action` and
    writes it as MP4 (imageio + ffmpeg), falling back to GIF if ffmpeg is missing.

    render_scale (default 3): frames come from HiResEpisodeRenderer - trails, border,
    items and head dots are REDRAWN vectorially at render_scale x engine resolution
    (2x supersampled internally), i.e. genuinely sharp lines instead of upscaled
    grid pixels. render_scale=None falls back to the legacy path (engine raster,
    nearest-neighbor upscaled by `scale`). Frames are STREAMED into the encoder -
    the old collect-everything-first approach held up to ~3 GB of raw frames in RAM
    for a 60s recording, which is exactly the kind of thing that kills a Colab
    session mid-training. Returns the written path.
    """
    from PIL import Image

    from ai.env import renderer

    s = env.constants.engine_resolution

    obs, _ = env.reset()
    hires: HiResEpisodeRenderer | None = None
    if render_scale is not None and render_scale >= 1:
        try:
            hires = HiResEpisodeRenderer(env.engine, render_scale)
        except Exception as exc:  # never let presentation kill the recording
            print(f"[video] hi-res renderer unavailable, falling back to raster upscale: {exc!r}")

    nn_size = s * max(1, scale)

    def grab() -> np.ndarray:
        if hires is not None:
            return hires.frame(env.engine)
        frame = renderer.render_frame(env.engine, s)
        if scale > 1:
            frame = np.asarray(Image.fromarray(frame).resize((nn_size, nn_size), Image.Resampling.NEAREST))
        return frame

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    gif_frames: list[np.ndarray] | None = None
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=7)
    except Exception:
        gif_frames = []  # no ffmpeg - collect (every 2nd frame later) for the GIF fallback

    def emit(frame: np.ndarray) -> None:
        if writer is not None:
            writer.append_data(frame)
        else:
            gif_frames.append(frame)

    last_frame = grab()
    emit(last_frame)
    terminated = truncated = False
    info: dict = {}
    ticks = 0
    for _ in range(max_ticks):
        obs, _r, terminated, truncated, info = env.step(int(np.asarray(predict_fn(obs)).reshape(-1)[0]))
        if hires is not None:
            hires.update(env.engine)
        last_frame = grab()
        emit(last_frame)
        ticks += 1
        if terminated or truncated:
            break

    # Warum ist das Video zu Ende? Ohne diese Einblendung sieht JEDES Videoende
    # wie ein Tod aus (eingefrorener letzter Frame) - auch wenn der Agent in
    # Wahrheit nur das Aufnahme- oder Episodenlimit erreicht hat und weiterlebt.
    if terminated and info.get("won"):
        end_reason = "won"  # Phasen mit Gegnern: alle anderen tot, Held lebt
        caption = f"GEWONNEN  (Tick {ticks})"
    elif terminated:
        end_reason = "death"
        cause = info.get("death_cause", "?")
        caption = f"GESTORBEN: {cause}  (Tick {ticks})"
    elif truncated:
        end_reason = "episode_limit"
        caption = f"EPISODEN-LIMIT ERREICHT - {ticks} Ticks ueberlebt"
    else:
        end_reason = "recording_limit"
        caption = f"AUFNAHMELIMIT ERREICHT - AGENT LEBT NOCH ({ticks}+ Ticks)"

    end_card = _annotate_frame(last_frame, caption)
    for _ in range(int(fps * 1.0)):  # hold the annotated final frame for a second so the outcome is readable
        emit(end_card)

    # maschinenlesbares Sidecar (step_X.json neben step_X.mp4): Ende-Grund,
    # Todesursache und überlebte Ticks - für spätere Auswertung ohne Videoschauen
    sidecar = {"end_reason": end_reason, "death_cause": info.get("death_cause"), "ticks_survived": ticks, "max_ticks": max_ticks}
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    if writer is not None:
        writer.close()
        return out_path
    # no ffmpeg available - a GIF (every 2nd frame, i.e. 30fps equivalent) still shows the behavior
    gif_path = out_path.with_suffix(".gif")
    imgs = [Image.fromarray(f) for f in gif_frames[::2]]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=1000 // 30, loop=0)
    return gif_path


def _annotate_frame(frame: "np.ndarray", caption: str) -> "np.ndarray":
    """Legt unten eine dunkle Leiste mit `caption` über eine Kopie des Frames."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(img)
    bar_h = max(24, img.height // 14)
    draw.rectangle([(0, img.height - bar_h), (img.width, img.height)], fill=(0, 0, 0))
    try:
        font = ImageFont.load_default(size=int(bar_h * 0.55))  # Pillow >= 10.1
    except TypeError:
        font = ImageFont.load_default()
    text_w = draw.textlength(caption, font=font)
    draw.text(((img.width - text_w) / 2, img.height - bar_h + bar_h * 0.2), caption, fill=(255, 255, 0), font=font)
    return np.asarray(img)


# --------------------------------------------------------------- best weights

_BEST_MODEL = "best_model.zip"
_BEST_META = "best_model.json"
_BEST_HISTORY = "best_history.csv"


class BestTracker:
    """Hält die BESTEN Gewichte eines Laufs fest, gemessen an einem gleitenden
    Episoden-Mittelwert (z. B. Überlebenszeit). Anders als die rotierenden
    Checkpoints (immer der NEUESTE Stand) überlebt hier der beste Stand - wichtig,
    weil PPO-Läufe zwischenzeitlich einbrechen können: endet das Training in
    einem Tal, ist best_model.zip trotzdem noch die Spitzen-Policy von früher.

    Resume-sicher: der Bestwert wird aus best_model.json wiedergeladen, ein
    fortgesetzter Lauf kann den historischen Bestwert also nie mit etwas
    Schlechterem überschreiben. Torch-frei testbar: `model` braucht nur .save().
    """

    def __init__(self, best_dir: str | Path, metric: str = "ep_len_mean", min_episodes: int = 50, min_delta: float = 0.0):
        self.best_dir = Path(best_dir)
        self.metric = metric
        self.min_episodes = min_episodes
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.best_step: int | None = None
        meta_path = self.best_dir / _BEST_META
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("metric") == metric:
                self.best_value = float(meta["value"])
                self.best_step = int(meta["step"])

    def maybe_save(self, model, mean_value: float, n_episodes: int, step: int) -> Path | None:
        """Speichert atomar, wenn `mean_value` (über >= min_episodes Episoden) den
        Bestwert um mehr als min_delta übertrifft. Gibt den Pfad oder None zurück."""
        if n_episodes < self.min_episodes:
            return None  # Burn-in: ein "Bestwert" aus 3 Glücks-Episoden wäre wertlos
        if self.best_value is not None and mean_value <= self.best_value + self.min_delta:
            return None

        self.best_dir.mkdir(parents=True, exist_ok=True)
        final = self.best_dir / _BEST_MODEL
        tmp = final.with_name(final.name + ".part")
        model.save(str(tmp))
        os.replace(tmp, final)  # gleiche Schreibgarantie wie die rotierenden Checkpoints

        self.best_value, self.best_step = float(mean_value), int(step)
        (self.best_dir / _BEST_META).write_text(
            json.dumps({"metric": self.metric, "value": self.best_value, "step": self.best_step, "n_episodes": n_episodes}, indent=2),
            encoding="utf-8",
        )
        history = self.best_dir / _BEST_HISTORY
        write_header = not history.exists()
        with open(history, "a", encoding="utf-8") as f:
            if write_header:
                f.write("step,metric,value,n_episodes\n")
            f.write(f"{step},{self.metric},{mean_value},{n_episodes}\n")
        return final
