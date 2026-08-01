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


def record_episode_video(env, predict_fn, out_path: str | Path, max_ticks: int = 1800, fps: int = 60, scale: int = 2):
    """Plays ONE episode in `env` (a CurveEnv) with `predict_fn(obs) -> action` and
    writes it as MP4 (imageio + ffmpeg), falling back to GIF if ffmpeg is missing.
    Frames are rendered at the engine's native resolution and nearest-neighbor
    upscaled - crisp 1px trails, no smoothing artifacts. Returns the written path.
    """
    from PIL import Image

    from ai.env import renderer

    s = env.constants.engine_resolution
    size = s * max(1, scale)

    def grab() -> np.ndarray:
        frame = renderer.render_frame(env.engine, s)
        if scale > 1:
            frame = np.asarray(Image.fromarray(frame).resize((size, size), Image.Resampling.NEAREST))
        return frame

    frames: list[np.ndarray] = []
    obs, _ = env.reset()
    frames.append(grab())
    terminated = truncated = False
    info: dict = {}
    ticks = 0
    for _ in range(max_ticks):
        obs, _r, terminated, truncated, info = env.step(int(predict_fn(obs)))
        frames.append(grab())
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

    end_card = _annotate_frame(frames[-1], caption)
    for _ in range(int(fps * 1.0)):  # hold the annotated final frame for a second so the outcome is readable
        frames.append(end_card)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # maschinenlesbares Sidecar (step_X.json neben step_X.mp4): Ende-Grund,
    # Todesursache und überlebte Ticks - für spätere Auswertung ohne Videoschauen
    sidecar = {"end_reason": end_reason, "death_cause": info.get("death_cause"), "ticks_survived": ticks, "max_ticks": max_ticks}
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=7) as writer:
            for f in frames:
                writer.append_data(f)
        return out_path
    except Exception:
        # no ffmpeg available - a GIF (every 2nd frame, i.e. 30fps equivalent) still shows the behavior
        gif_path = out_path.with_suffix(".gif")
        imgs = [Image.fromarray(f) for f in frames[::2]]
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
