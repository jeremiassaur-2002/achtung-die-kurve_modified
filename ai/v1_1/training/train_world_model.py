"""Offline-Training des Weltmodells auf den Planer-Trajektorien.

    python -m ai.run train --version v1_1 --config ai/v1_1/config/dreamer.yaml

Rein offline: die Engine wird waehrend des Trainings kein einziges Mal
angefasst. Das ist der Punkt, an dem "statisch trainieren" konkret wird - und
der Grund, warum diese Phase auf einer GPU-Maschine ohne viele CPU-Kerne
laeuft, waehrend das Sammeln auf billigen Kernen passiert.

**Woran man erkennt, ob es funktioniert.** Nicht am Gesamtverlust - der wird vom
Bildterm dominiert (12288 Pixel gegen einen Belohnungswert) und sinkt auch dann,
wenn das Modell nur den schwarzen Hintergrund gelernt hat. Die aussagekraeftigen
Groessen stehen einzeln im Log:

  `reward_mae`    mittlerer Belohnungsfehler. Muss deutlich unter 0,1 fallen,
                  sonst kann der Critic spaeter nichts damit anfangen.
  `continue_acc`  Trefferquote der Fortsetzungsvorhersage. Der Tod ist selten;
                  ein Modell, das immer "geht weiter" sagt, kommt hier schon
                  ueber 99% - deshalb zusaetzlich `dream_*` beobachten.
  `kl_dyn`        wie weit Prior und Posterior auseinanderliegen. Faellt der
                  Wert auf die free nats (1,0) und bleibt dort, ist alles gut.
                  Faellt er auf ~0, ist der Posterior kollabiert und der Traum
                  wird nutzlos.
  `dream_vec_mae` der eigentliche Test (siehe `--dream-every`): Kontext
                  einspielen, dann N Ticks blind weitertraeumen und die
                  getraeumten Sensorwerte gegen die echten halten. Das misst,
                  was die Imagination taugt - unabhaengig von jeder Policy.

**Checkpoints** rotieren wie in v1_0: erst vollstaendig in eine temporaere Datei
schreiben, dann umbenennen, dann alte loeschen. Ein Absturz mitten im Speichern
darf nie den letzten brauchbaren Stand zerstoeren.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ai.core.utils.paths import RunPaths
from ai.core.utils.timing import PhaseTimer
from ai.v1_1.data.replay import Batch, ShardReplay
from ai.v1_1.models.world_model import WorldModel, WorldModelConfig


def to_torch(batch: Batch, device) -> dict[str, torch.Tensor]:
    return {
        "image": torch.from_numpy(batch.image).to(device),
        "vector": torch.from_numpy(batch.vector).to(device),
        "action": torch.from_numpy(batch.action).to(device),
        "reward": torch.from_numpy(batch.reward).to(device),
        "is_first": torch.from_numpy(batch.is_first).to(device),
        "is_terminal": torch.from_numpy(batch.is_terminal).to(device),
        "mask": torch.from_numpy(batch.mask).to(device),
        "ticks_to_death": torch.from_numpy(batch.ticks_to_death).to(device),
    }


def build_model(cfg: dict, replay: ShardReplay) -> WorldModel:
    """Eingangsgroessen kommen aus den DATEN, nicht aus der Config: wurde der
    Datensatz mit 16 Strahlen gesammelt und die Config seither auf 36 gestellt,
    soll das Modell zum Datensatz passen - und der Widerspruch soll auffallen,
    statt sich in einem Shape-Fehler tief im Encoder zu aeussern."""
    shapes = replay.shapes()
    wm_cfg = cfg.get("world_model", {})
    want_v = cfg.get("n_rays")
    got_v = shapes["vector"][0]
    if want_v is not None and got_v != 16 + 6 + want_v + 3:
        print(
            f"[wm] Hinweis: Datensatz hat Vektorlaenge {got_v}, die Config impliziert "
            f"{16 + 6 + want_v + 3} (n_rays={want_v}). Das Modell folgt dem Datensatz. "
            f"Wenn das nicht gewollt ist: neu sammeln."
        )
    return WorldModel(
        WorldModelConfig(
            image_shape=shapes["image"],
            vector_dim=got_v,
            action_dim=shapes["action"][0],
            deter_dim=wm_cfg.get("deter_dim", 512),
            stoch_groups=wm_cfg.get("stoch_dim", 32),
            stoch_classes=wm_cfg.get("stoch_classes", 32),
            hidden_dim=wm_cfg.get("hidden_dim", 512),
            cnn_depth=wm_cfg.get("cnn_depth", 32),
            kl_free_nats=wm_cfg.get("kl_free_nats", 1.0),
            continue_pos_weight=wm_cfg.get("continue_pos_weight", 50.0),
            ttd_cap=wm_cfg.get("ttd_cap", 120.0),
            survival_weight=wm_cfg.get("survival_weight", 1.0),
            kl_beta_dyn=wm_cfg.get("kl_beta_dyn", 0.5),
            kl_beta_rep=wm_cfg.get("kl_beta_rep", 0.1),
        )
    )


@torch.no_grad()
def dream_diagnostic(model: WorldModel, replay: ShardReplay, device, context: int = 8, horizon: int = 32) -> dict:
    """Kontext einspielen, dann blind weitertraeumen und mit der Wirklichkeit
    vergleichen. Der ehrlichste verfuegbare Test des Weltmodells."""
    model.eval()
    batch = to_torch(replay.sample(8, context + horizon), device)
    post, _ = model.observe(
        batch["image"][:, :context], batch["vector"][:, :context], batch["action"][:, :context], batch["is_first"][:, :context]
    )
    start = post[:, -1]
    dreamed = model.dream(start, batch["action"][:, context : context + horizon])
    real_vec = batch["vector"][:, context : context + horizon]
    mask = batch["mask"][:, context : context + horizon]
    denom = mask.sum().clamp(min=1.0)

    from ai.v1_1.models.nets import symlog

    vec_mae = ((dreamed["vector"] - symlog(real_vec)).abs().mean(-1) * mask).sum() / denom
    real_img = batch["image"][:, context : context + horizon].float()
    img_mae = ((dreamed["image"].float() - real_img).abs().mean(dim=(2, 3, 4)) * mask).sum() / denom
    model.train()
    return {
        "dream_vec_mae": float(vec_mae),
        "dream_img_mae": float(img_mae),
        "dream_horizon": horizon,
    }


def _save_checkpoint(path: Path, model: WorldModel, opt, step: int, keep: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "model": model.state_dict(), "optim": opt.state_dict(), "cfg": asdict(model.cfg)}, tmp)
    tmp.replace(path)
    olds = sorted(path.parent.glob("wm_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in olds[keep:]:
        old.unlink(missing_ok=True)


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    cands = sorted(ckpt_dir.glob("wm_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def train_world_model(
    cfg: dict, paths: RunPaths, timer: PhaseTimer, dataset_dir: Path, resume: str | None = "auto"
) -> Path:
    wm_cfg = cfg.get("world_model", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay = ShardReplay(
        dataset_dir,
        cache_episodes=wm_cfg.get("cache_episodes", 64),
        seed=cfg.get("seed", 0),
        terminal_fraction=wm_cfg.get("terminal_fraction", 0.25),
        ttd_cap=wm_cfg.get("ttd_cap", 120.0),
        split="train",
        val_fraction=wm_cfg.get("val_fraction", 0.1),
    )
    val_replay = ShardReplay(
        dataset_dir, cache_episodes=8, seed=cfg.get("seed", 0) + 1,
        terminal_fraction=wm_cfg.get("terminal_fraction", 0.25),
        ttd_cap=wm_cfg.get("ttd_cap", 120.0), split="val", val_fraction=wm_cfg.get("val_fraction", 0.1),
    )
    print(f"[wm] Train: {json.dumps(replay.stats())}")
    print(f"[wm] Val:   {json.dumps(val_replay.stats())}")
    if val_replay.total_ticks < 5 * wm_cfg.get("batch_length", 64):
        print(
            f"[wm] WARNUNG: Val-Split hat nur {val_replay.total_ticks:,} Ticks. Die Validierungszahlen "
            f"schwanken damit so stark, dass sie nichts aussagen. Mehr Episoden sammeln oder "
            f"val_fraction erhoehen."
        )
    if not replay._fatal:
        print(
            "[wm] WARNUNG: keine einzige Episode endet mit einem Tod. Der Fortsetzungs- und "
            "der Restlebensdauer-Kopf koennen so nichts lernen, und der Actor wird in der "
            "Imagination nie sterben. Mehr Daten sammeln oder planner.epsilon erhoehen."
        )

    model = build_model(cfg, replay).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[wm] {n_params / 1e6:.1f}M Parameter auf {device}")

    opt = torch.optim.Adam(model.parameters(), lr=wm_cfg.get("learning_rate", 1e-4), eps=1e-8)
    start_step = 0
    if resume not in (None, "none"):
        ckpt = Path(resume) if resume != "auto" else latest_checkpoint(paths.checkpoints)
        if ckpt and ckpt.exists():
            state = torch.load(ckpt, map_location=device, weights_only=False)
            try:
                model.load_state_dict(state["model"])
            except RuntimeError as e:
                raise SystemExit(
                    f"[wm] {ckpt} passt nicht zur aktuellen Modellform ({e}).\n"
                    f"     Das passiert, wenn sich Beobachtungsgroesse oder Modellbreite geaendert hat. "
                    f"Ein Fortsetzen ist dann nicht moeglich - neuer --run-name."
                ) from e
            opt.load_state_dict(state["optim"])
            start_step = int(state["step"])
            print(f"[wm] fortgesetzt ab Schritt {start_step:,}")

    batch_size = wm_cfg.get("batch_size", 16)
    batch_length = wm_cfg.get("batch_length", 64)
    total_steps = wm_cfg.get("train_steps", 100_000)
    log_every = wm_cfg.get("log_every_steps", 200)
    ckpt_every = cfg.get("checkpoint_every_steps", 5000)
    dream_every = wm_cfg.get("dream_every_steps", 2000)
    grad_clip = wm_cfg.get("grad_clip", 1000.0)

    metrics_path = paths.metrics / "world_model.csv"
    if not metrics_path.exists():
        metrics_path.write_text("step,total,image,vector,reward,continue,kl,kl_dyn,reward_mae,continue_acc\n")

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(paths.tensorboard))
    except ImportError:
        print("[wm] tensorboard nicht installiert - nur CSV-Logging")

    model.train()
    t_last = time.perf_counter()
    with timer.phase("world_model", steps=total_steps, params=n_params, device=str(device)):
        for step in range(start_step, total_steps):
            with timer.section("wm_sample"):
                batch = to_torch(replay.sample(batch_size, batch_length), device)
            with timer.section("wm_update"):
                loss, _ = model.loss(**batch)
                opt.zero_grad(set_to_none=True)
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            if step % log_every == 0:
                p = loss.parts
                loss_val = float(loss.total.detach())
                dt = time.perf_counter() - t_last
                sps = log_every / dt if step > start_step else 0.0
                t_last = time.perf_counter()
                print(
                    f"[wm] {step:>7,}/{total_steps:,} | total {loss_val:9.2f} | "
                    f"img {p['image']:8.2f} vec {p['vector']:6.2f} | rew_mae {p['reward_mae']:.4f} | "
                    f"term_recall {p['terminal_recall']:.3f} | ttd_mae {p['survival_mae']:.1f} | "
                    f"kl_dyn {p['kl_dyn']:.3f} | {sps:.1f} it/s"
                )
                with metrics_path.open("a") as f:
                    f.write(
                        f"{step},{loss_val:.4f},{p['image']:.4f},{p['vector']:.4f},{p['reward']:.4f},"
                        f"{p['continue']:.4f},{p['kl']:.4f},{p['kl_dyn']:.4f},{p['reward_mae']:.5f},{p['continue_acc']:.4f}\n"
                    )
                if writer:
                    writer.add_scalar("wm/total", loss_val, step)
                    for k, v in p.items():
                        writer.add_scalar(f"wm/{k}", v, step)

            if dream_every and step > start_step and step % dream_every == 0:
                with timer.section("wm_dream_eval"):
                    # Auf dem VAL-Split: die Traumguete auf Trainingsdaten misst
                    # auch Auswendiglernen, und genau das will man hier nicht.
                    d = dream_diagnostic(model, val_replay, device, horizon=wm_cfg.get("dream_horizon", 32))
                    with torch.no_grad():
                        vb = to_torch(val_replay.sample(batch_size, batch_length), device)
                        vloss, _ = model.loss(**vb)
                    d["val_total"] = float(vloss.total)
                    d["val_terminal_recall"] = vloss.parts["terminal_recall"]
                    d["val_survival_mae"] = vloss.parts["survival_mae"]
                print(f"[wm] Traum-Diagnose @ {step:,}: {d}")
                if writer:
                    for k, v in d.items():
                        writer.add_scalar(f"dream/{k}", v, step)

            if ckpt_every and step > start_step and step % ckpt_every == 0:
                _save_checkpoint(paths.checkpoints / f"wm_{step:08d}.pt", model, opt, step, cfg.get("checkpoint_keep", 2))

    final = paths.best / "world_model.pt"
    final.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": total_steps, "model": model.state_dict(), "cfg": asdict(model.cfg)}, final)
    if writer:
        writer.close()
    print(f"[wm] fertig -> {final}")
    return final
