"""Actor/Critic in der Imagination des Weltmodells trainieren.

Ein Schritt:

  1. Echte Sequenz aus dem Replay -> Weltmodell rollt aus -> Posterior-Zustaende
  2. Diese Zustaende (alle, flach) sind die Startpunkte fuer den Traum
  3. `imagination_horizon` Schritte traeumen: Actor waehlt, RSSM-Prior rechnet
     weiter, Belohnungs- und Fortsetzungskopf bewerten
  4. lambda-Renditen darauf, Critic lernt sie vorherzusagen, Actor lernt per
     REINFORCE mit dem Critic als Baseline

Das Weltmodell ist dabei EINGEFROREN. Zwei Optimierer, die gleichzeitig an
denselben Gewichten ziehen, waeren ein bewegliches Ziel: der Actor optimiert
gegen ein Modell, das sich unter ihm veraendert, und beide koennen gemeinsam in
eine Fantasie abdriften, in der es sich bequem lebt. Getrennte Phasen sind
langsamer, aber ehrlich - und passen zum offline gesammelten Datensatz.

**BC-Warmstart.** Vor dem Imaginationstraining wird der Actor auf die
Planer-Aktionen im Datensatz geklont. Ohne das startet er gleichverteilt, stirbt
in den ersten getraeumten Schritten und lernt aus lauter kurzen, gleich
schlechten Trajektorien fast nichts. Mit Warmstart beginnt die Imagination in
einer Region, die der Planer schon als ueberlebensfaehig erwiesen hat - dieselbe
Ueberlegung wie beim BC-Kickstart in v1_0, nur ist der Experte hier die
Beam-Suche statt einer frueheren Policy.

**KL-Anker zur BC-Policy (`bc_kl_scale`).** Das klassische Problem des
Offline-Lernens: der Actor optimiert gegen ein Weltmodell, das nur die
Zustaende kennt, die der Planer besucht hat. Verlaesst er diese Region, sagt
das Modell dort irgendetwas vorher - oft etwas zu Optimistisches, denn nichts
hat ihm je widersprochen - und der Actor lernt begeistert, genau dorthin zu
laufen. Im Traum sieht das nach Fortschritt aus, in der Engine stirbt er.
Gegenmittel: eine eingefrorene Kopie des BC-Actors, und ein KL-Term, der die
Policy in ihrer Naehe haelt. Der Anker begrenzt bewusst, wie weit der Actor sich
vom Planer entfernen darf - `bc_kl_scale` ist der Regler dafuer. 0 schaltet ihn
ab, was nur mit online nachgesammelten Daten sinnvoll ist.

**Warum das nicht einfach nur BC ist.** Verhaltensklonung kopiert den Planer,
Fehler eingeschlossen, und kann ihn nie uebertreffen. Der Planer sieht ein
eingefrorenes Gitter und kann nicht antizipieren, was Gegner als naechstes
zeichnen. Genau das kann ein Weltmodell lernen - und der Actor kann es in der
Imagination ausnutzen. BC liefert den Startpunkt, die Imagination den Fortschritt
darueber hinaus.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ai.core.utils.paths import RunPaths
from ai.core.utils.timing import PhaseTimer
from ai.v1_1.data.replay import ShardReplay
from ai.v1_1.models.actor_critic import Actor, ActorCriticConfig, Critic, ReturnNormalizer, lambda_return
from ai.v1_1.models.world_model import WorldModel, WorldModelConfig
from ai.v1_1.training.train_world_model import to_torch


def load_world_model(path: Path, device) -> WorldModel:
    state = torch.load(path, map_location=device, weights_only=False)
    model = WorldModel(WorldModelConfig(**{**state["cfg"], "image_shape": tuple(state["cfg"]["image_shape"])}))
    model.load_state_dict(state["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def imagine_rollout(
    world_model: WorldModel, actor: Actor, start_feat_state, horizon: int
) -> dict[str, torch.Tensor]:
    """Aus (B, ...) Startzustaenden `horizon` Schritte traeumen.

    Der Gradient des Actors muss durch die Aktionen fliessen, der des
    Weltmodells nicht - daher no_grad um die Modellkoepfe, aber log_probs
    ausserhalb gesammelt.
    """
    rssm = world_model.rssm
    state = start_feat_state
    feats, log_probs, entropies = [], [], []
    for _ in range(horizon):
        feat = state.feature
        dist = actor.distribution(feat)
        idx = dist.sample()
        log_probs.append(dist.log_prob(idx))
        entropies.append(dist.entropy())
        action = F.one_hot(idx, actor.cfg.action_dim).float()
        with torch.no_grad():
            state = rssm.img_step(state, action)
        feats.append(state.feature)
    feat_seq = torch.stack(feats, dim=1)  # (B, H, F)
    flat = feat_seq.reshape(-1, feat_seq.shape[-1])
    with torch.no_grad():
        world_model.twohot.to(flat.device)
        reward = world_model.twohot.decode(world_model.reward_head(flat)).view(feat_seq.shape[:2])
        cont = torch.sigmoid(world_model.continue_head(flat)).view(feat_seq.shape[:2])
        # Die getraeumte Restlebensdauer ist die aussagekraeftigste Diagnose
        # WAEHREND des Policy-Trainings: faellt sie, laeuft der Actor in
        # gefaehrlichere Zustaende, obwohl seine Rendite steigt - das sichere
        # Zeichen dafuer, dass er ein Leck im Weltmodell ausnutzt.
        ttd = world_model.twohot.decode(world_model.survival_head(flat)).view(feat_seq.shape[:2])
    return {
        "feat": feat_seq,
        "reward": reward,
        "continue": cont,
        "ticks_to_death": ttd,
        "log_prob": torch.stack(log_probs, dim=1),
        "entropy": torch.stack(entropies, dim=1),
    }


def bc_warmstart(
    world_model: WorldModel, actor: Actor, replay: ShardReplay, device, steps: int, batch_size: int, batch_length: int, lr: float
) -> float:
    """Actor auf die Planer-Aktionen klonen. Liefert die erreichte Trefferquote."""
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    acc = 0.0
    for step in range(steps):
        batch = to_torch(replay.sample(batch_size, batch_length), device)
        with torch.no_grad():
            post, _ = world_model.observe(batch["image"], batch["vector"], batch["action"], batch["is_first"])
            feat = post.feature
        logits = actor.logits(feat)
        target = batch["action"].argmax(-1)
        mask = batch["mask"]
        loss = (F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none").view_as(mask) * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                acc = float((((logits.argmax(-1) == target).float()) * mask).sum() / mask.sum().clamp(min=1))
            print(f"[ac] BC {step:>6,}/{steps:,} | loss {float(loss.detach()):.4f} | Trefferquote {acc:.3f}")
    return acc


def train_policy(
    cfg: dict, paths: RunPaths, timer: PhaseTimer, dataset_dir: Path, world_model_path: Path, resume: str | None = "auto"
) -> Path:
    pcfg = cfg.get("policy", {})
    wm_cfg = cfg.get("world_model", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay = ShardReplay(dataset_dir, cache_episodes=wm_cfg.get("cache_episodes", 64), seed=cfg.get("seed", 0))
    world_model = load_world_model(world_model_path, device)
    ac_cfg = ActorCriticConfig(
        action_dim=world_model.cfg.action_dim,
        hidden_dim=pcfg.get("hidden_dim", 512),
        imagination_horizon=pcfg.get("imagination_horizon", 15),
        gamma=pcfg.get("gamma", 0.997),
        lambda_=pcfg.get("lambda_", 0.95),
        entropy_scale=pcfg.get("entropy_scale", 3e-4),
        actor_lr=pcfg.get("actor_lr", 3e-5),
        critic_lr=pcfg.get("critic_lr", 3e-5),
        slow_critic_decay=pcfg.get("slow_critic_decay", 0.98),
    )
    actor = Actor(world_model.feat_dim, ac_cfg).to(device)
    critic = Critic(world_model.feat_dim, ac_cfg).to(device)
    print(f"[ac] Actor {sum(p.numel() for p in actor.parameters())/1e6:.2f}M | Critic {sum(p.numel() for p in critic.parameters())/1e6:.2f}M | {device}")

    start_step = 0
    ckpt_path = paths.checkpoints / "policy.pt"
    if resume not in (None, "none") and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        start_step = int(state["step"])
        print(f"[ac] fortgesetzt ab Schritt {start_step:,}")

    batch_size = pcfg.get("batch_size", 16)
    batch_length = pcfg.get("batch_length", 16)
    horizon = ac_cfg.imagination_horizon

    if start_step == 0 and pcfg.get("bc_steps", 500) > 0:
        with timer.phase("policy_bc", steps=pcfg.get("bc_steps", 500)):
            acc = bc_warmstart(
                world_model, actor, replay, device,
                steps=pcfg.get("bc_steps", 500),
                batch_size=batch_size, batch_length=batch_length,
                lr=pcfg.get("bc_lr", 3e-4),
            )
        print(f"[ac] BC-Warmstart fertig, Trefferquote {acc:.3f}")

    # Eingefrorener Zwilling nach dem BC-Warmstart als Referenzverteilung.
    bc_kl_scale = float(pcfg.get("bc_kl_scale", 0.1))
    bc_actor = None
    if bc_kl_scale > 0.0:
        import copy

        bc_actor = copy.deepcopy(actor).eval()
        for p_ in bc_actor.parameters():
            p_.requires_grad_(False)
        print(f"[ac] KL-Anker zur BC-Policy aktiv (bc_kl_scale={bc_kl_scale})")

    opt_actor = torch.optim.Adam(actor.parameters(), lr=ac_cfg.actor_lr)
    opt_critic = torch.optim.Adam(critic.net.parameters(), lr=ac_cfg.critic_lr)
    normalizer = ReturnNormalizer(decay=ac_cfg.return_norm_decay)

    total_steps = pcfg.get("train_steps", 50_000)
    log_every = pcfg.get("log_every_steps", 200)
    ckpt_every = cfg.get("checkpoint_every_steps", 5000)

    metrics_path = paths.metrics / "policy.csv"
    if not metrics_path.exists():
        metrics_path.write_text(
            "step,actor_loss,critic_loss,entropy,return_mean,value_mean,imag_reward,imag_continue,imag_ttd,bc_kl\n"
        )
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(paths.tensorboard))
    except ImportError:
        pass

    t_last = time.perf_counter()
    with timer.phase("policy", steps=total_steps, horizon=horizon, device=str(device)):
        for step in range(start_step, total_steps):
            with timer.section("ac_encode"):
                batch = to_torch(replay.sample(batch_size, batch_length), device)
                with torch.no_grad():
                    post, _ = world_model.observe(batch["image"], batch["vector"], batch["action"], batch["is_first"])
                # Alle Zeitschritte als Startpunkte: das vervielfacht die
                # Startzustaende ohne zusaetzliche Encoder-Arbeit. Padding wird
                # verworfen, sonst startete der Traum in schwarzen
                # Platzhalterbildern.
                keep = batch["mask"].reshape(-1) > 0
                flat_state = post[
                    (
                        torch.arange(batch_size, device=device).repeat_interleave(batch_length)[keep],
                        torch.arange(batch_length, device=device).repeat(batch_size)[keep],
                    )
                ].detach()

            with timer.section("ac_imagine"):
                roll = imagine_rollout(world_model, actor, flat_state, horizon)

            feat = roll["feat"]
            with torch.no_grad():
                target_value = critic.slow_value(feat)
                returns = lambda_return(roll["reward"], target_value, roll["continue"], ac_cfg.gamma, ac_cfg.lambda_)
                scale = normalizer.update(returns)

            # --- Critic: die Renditen vorhersagen ---
            with timer.section("ac_critic"):
                v_logits = critic.logits(feat.detach())
                critic_loss = critic.twohot.to(device).loss(v_logits, returns).mean()
                opt_critic.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.net.parameters(), 100.0)
                opt_critic.step()
                critic.update_slow()

            # --- Actor: REINFORCE mit Critic-Baseline ---
            with timer.section("ac_actor"):
                with torch.no_grad():
                    baseline = critic.value(feat)
                    advantage = (returns - baseline) / scale
                actor_loss = -(roll["log_prob"] * advantage).mean() - ac_cfg.entropy_scale * roll["entropy"].mean()
                bc_kl = torch.zeros((), device=device)
                if bc_actor is not None:
                    cur = torch.log_softmax(actor.logits(feat.detach()), -1)
                    with torch.no_grad():
                        ref = torch.softmax(bc_actor.logits(feat.detach()), -1)
                    bc_kl = (ref * (torch.log(ref.clamp(min=1e-8)) - cur)).sum(-1).mean()
                    actor_loss = actor_loss + bc_kl_scale * bc_kl
                opt_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
                opt_actor.step()

            if step % log_every == 0:
                dt = time.perf_counter() - t_last
                sps = log_every / dt if step > start_step else 0.0
                t_last = time.perf_counter()
                ent = float(roll["entropy"].mean())
                ret = float(returns.mean())
                print(
                    f"[ac] {step:>7,}/{total_steps:,} | actor {float(actor_loss):+.4f} critic {float(critic_loss):.4f} | "
                    f"Entropie {ent:.3f} | Rendite {ret:+.3f} | Traum-cont {float(roll['continue'].mean()):.3f} | "
                    f"Traum-ttd {float(roll['ticks_to_death'].mean()):.1f} | BC-KL {float(bc_kl):.4f} | {sps:.1f} it/s"
                )
                with metrics_path.open("a") as f:
                    f.write(
                        f"{step},{float(actor_loss):.5f},{float(critic_loss):.5f},{ent:.5f},{ret:.5f},"
                        f"{float(baseline.mean()):.5f},{float(roll['reward'].mean()):.5f},{float(roll['continue'].mean()):.5f},"
                        f"{float(roll['ticks_to_death'].mean()):.3f},{float(bc_kl):.5f}\n"
                    )
                if writer:
                    writer.add_scalar("ac/actor_loss", float(actor_loss), step)
                    writer.add_scalar("ac/critic_loss", float(critic_loss), step)
                    writer.add_scalar("ac/entropy", ent, step)
                    writer.add_scalar("ac/return", ret, step)
                    writer.add_scalar("ac/imag_continue", float(roll["continue"].mean()), step)
                    writer.add_scalar("ac/imag_ttd", float(roll["ticks_to_death"].mean()), step)
                    writer.add_scalar("ac/bc_kl", float(bc_kl), step)

            if ckpt_every and step > start_step and step % ckpt_every == 0:
                tmp = ckpt_path.with_suffix(".tmp")
                torch.save({"step": step, "actor": actor.state_dict(), "critic": critic.state_dict()}, tmp)
                tmp.replace(ckpt_path)

    final = paths.best / "policy.pt"
    final.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": total_steps,
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "ac_cfg": ac_cfg.__dict__,
            "world_model": str(world_model_path),
        },
        final,
    )
    if writer:
        writer.close()
    print(f"[ac] fertig -> {final}")
    return final
