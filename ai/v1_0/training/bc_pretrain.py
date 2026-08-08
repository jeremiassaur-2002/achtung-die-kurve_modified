"""Behavior-cloning warm start: teach the policy the rule-based bot's behavior
SUPERVISED first, then fine-tune with PPO via `train.py --init-from`.

Why this exists: PPO from scratch spends its first millions of steps discovering
that walls and trails kill - with a sparse death signal and a 3-action space that
looks locally symmetric. The heuristic bot (ai/core/env/rules_bot.py, "hard" survives
thousands of ticks and never clips its own line) already embodies exactly the
survival instincts Phase 1 is trying to learn. Cloning it skips the
random-flailing phase entirely and hands PPO a policy that already avoids walls,
avoids trails, and commits to turns - PPO then only has to IMPROVE on it.

Two stages, one command:

  python -m ai.v1_0.training.bc_pretrain --config ai/v1_0/config/phase1.yaml \
      --out ai/runs/bc/bc_phase1.zip

  python -m ai.v1_0.training.train --config ai/v1_0/config/phase1.yaml \
      --init-from ai/runs/bc/bc_phase1.zip

Design decisions:
  * Data goes through the REAL CurveEnv (make_configs from train.py): identical
    frame stacking, sensors, reward semantics and curriculum stage-0 opponent mix
    as the later fine-tune - any observation drift here would poison the warm
    start. The teacher drives the HERO seat; with probability --explore a random
    action is EXECUTED instead (wider state coverage - DAgger-style), but the
    LABEL is always what the teacher wanted in that state.
  * Both policy AND value head are trained: actions via negative log-likelihood
    (= cross-entropy) against the teacher, values via MSE against the discounted
    Monte-Carlo return of the phase's actual reward. A cloned policy with a
    garbage value head destabilizes the first PPO updates; regressing V toward
    real returns removes that cliff. Episodes cut by truncation (or by the buffer
    filling up) bootstrap with 0 - slightly pessimistic, documented, harmless.
  * A small entropy bonus keeps the cloned policy from going fully deterministic
    (pure CE loves 0/1 outputs; PPO needs some exploration mass to improve).
  * Storage is uint8/float32 np.memmap on disk, not RAM: 50k transitions of
    (12,96,96) uint8 stacks are ~5.5 GB - exactly what a Colab session cannot
    hold next to torch. Memmaps make dataset size a disk knob.
  * Only algo: ppo / maskable_ppo can be cloned this way (their policies expose
    evaluate_actions -> log_prob/values/entropy). The result is a full SB3 zip:
    resumable, exportable, --init-from-able like any checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from ai.core.env.curve_env import CurveEnv
from ai.core.env.rules_bot import BOT_DIFFICULTIES, RuleBasedBot
from ai.v1_0.models.algo import build_algo_kwargs, resolve_algo
from ai.v1_0.models.policy import build_policy_kwargs
from ai.v1_0.training.curriculum import CurriculumManager
from ai.v1_0.training.train import build_stage, load_config, make_configs

_ACTIONS = (0, 1, 2)


def _stage0_factory(cfg: dict, obs_cfg, seed: int):
    """The curriculum's FIRST stage, exactly as train.py would start it (self_play/
    league tokens fall back to random - BC is a pre-Phase warm start, there are no
    snapshots yet)."""
    curriculum_cfg = cfg["curriculum"]
    stages = [build_stage(s, None, None) for s in curriculum_cfg["stages"]]
    manager = CurriculumManager(
        stages,
        blend_window_episodes=curriculum_cfg.get("blend_window_episodes", 100),
        rolling_window=curriculum_cfg.get("rolling_window", 200),
        pool_variants=curriculum_cfg.get("pool_variants", 8),
        rng_seed=seed,
    )
    return manager.make_factory(obs_cfg, rng_seed=seed)


def generate_dataset(
    cfg: dict,
    data_dir: str | Path,
    n_transitions: int = 50_000,
    teacher: str = "hard",
    explore_eps: float = 0.05,
    seed: int = 0,
) -> dict:
    """Rolls teacher-driven episodes and writes (obs, teacher_action, return) as
    memmaps into `data_dir`. Returns the meta dict (also saved as meta.json)."""
    if teacher != "mix" and teacher not in BOT_DIFFICULTIES:
        raise SystemExit(f"[bc] unbekannter Lehrer {teacher!r} - erlaubt: {list(BOT_DIFFICULTIES)} oder 'mix'")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    obs_cfg, env_config = make_configs(cfg)
    env = CurveEnv(_stage0_factory(cfg, obs_cfg, seed), config=env_config, seed=seed)
    gamma = float(cfg.get("ppo", {}).get("gamma", 0.99))

    c, h, w = obs_cfg.image_shape
    from ai.core.env.observation import vector_dim

    d = vector_dim(obs_cfg)
    images = np.lib.format.open_memmap(data_dir / "images.npy", mode="w+", dtype=np.uint8, shape=(n_transitions, c, h, w))
    vectors = np.lib.format.open_memmap(data_dir / "vectors.npy", mode="w+", dtype=np.float32, shape=(n_transitions, d))
    actions = np.lib.format.open_memmap(data_dir / "actions.npy", mode="w+", dtype=np.uint8, shape=(n_transitions,))
    returns = np.lib.format.open_memmap(data_dir / "returns.npy", mode="w+", dtype=np.float32, shape=(n_transitions,))

    rng = random.Random(seed)
    filled = 0
    episodes = 0
    ep_ticks: list[int] = []
    t0 = time.time()
    mix_pool = ("medium", "hard", "hunter")  # teacher="mix": Vielfalt statt 14x derselbe Fahrstil
    while filled < n_transitions:
        obs, _ = env.reset()
        difficulty = rng.choice(mix_pool) if teacher == "mix" else teacher
        bot = RuleBasedBot(difficulty, random.Random(rng.randrange(2**31)))
        ep_start = filled
        ep_rewards: list[float] = []
        while filled < n_transitions:
            label = int(bot.decide(env.engine, env.hero_name))
            images[filled] = obs["image"]
            vectors[filled] = obs["vector"]
            actions[filled] = label
            executed = rng.choice(_ACTIONS) if rng.random() < explore_eps else label
            obs, reward, terminated, truncated, _info = env.step(executed)
            ep_rewards.append(float(reward))
            filled += 1
            if terminated or truncated:
                break
        # discounted Monte-Carlo returns for the value head; cut/truncated tails
        # bootstrap with 0 (see module docstring)
        g = 0.0
        for i in range(len(ep_rewards) - 1, -1, -1):
            g = ep_rewards[i] + gamma * g
            returns[ep_start + i] = g
        episodes += 1
        ep_ticks.append(len(ep_rewards))
        if episodes % 5 == 0 or filled >= n_transitions:
            rate = filled / max(1e-9, time.time() - t0)
            print(f"[bc] {filled:,}/{n_transitions:,} Transitionen | {episodes} Episoden | Ø {np.mean(ep_ticks):.0f} Ticks | {rate:.0f}/s")

    for arr in (images, vectors, actions, returns):
        arr.flush()
    meta = {
        "n_transitions": int(n_transitions),
        "teacher": teacher,
        "explore_eps": explore_eps,
        "gamma": gamma,
        "episodes": episodes,
        "mean_episode_ticks": float(np.mean(ep_ticks)),
        "image_shape": [c, h, w],
        "vector_dim": d,
        "seed": seed,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[bc] Datensatz fertig: {data_dir} (Lehrer '{teacher}' überlebt im Schnitt {meta['mean_episode_ticks']:.0f} Ticks)")
    return meta


def train_bc(
    cfg: dict,
    data_dir: str | Path,
    out_path: str | Path,
    epochs: int = 4,
    batch_size: int = 256,
    lr: float = 3e-4,
    value_coef: float = 0.5,
    ent_coef: float = 0.003,
    val_frac: float = 0.05,
    device: str = "auto",
    seed: int = 0,
) -> Path:
    import torch as th
    import torch.nn.functional as F

    from ai.core.env.vec_factory import build_vec_env

    spec = resolve_algo(cfg)
    if spec.name not in ("ppo", "maskable_ppo"):
        raise SystemExit(
            f"[bc] Behavior Cloning ist für algo: {spec.name} nicht implementiert - der Klon-Loss braucht "
            f"evaluate_actions (log_prob/values/entropy) einer Actor-Critic-Policy. Für den Kickstart algo: ppo "
            f"setzen; ein BC-PPO-Modell kann NICHT als --init-from für recurrent_ppo/qrdqn dienen (andere Familie)."
        )

    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    images = np.load(data_dir / "images.npy", mmap_mode="r")
    vectors = np.load(data_dir / "vectors.npy", mmap_mode="r")
    actions = np.load(data_dir / "actions.npy", mmap_mode="r")
    returns = np.load(data_dir / "returns.npy", mmap_mode="r")
    n = images.shape[0]

    obs_cfg, env_config = make_configs(cfg)
    vec_env = build_vec_env(
        n_envs=1, opponent_factory=_stage0_factory(cfg, obs_cfg, seed), config=env_config, base_seed=seed, use_subprocess=False
    )
    model = spec.cls(
        spec.policy_name,
        vec_env,
        policy_kwargs=build_policy_kwargs(cnn_arch=cfg.get("cnn_arch", "small"), algo=spec.name),
        verbose=0,
        seed=seed,
        **build_algo_kwargs(cfg, spec),
    )
    dev = model.policy.device
    optimizer = th.optim.Adam(model.policy.parameters(), lr=lr)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def batch_tensors(idx: np.ndarray):
        idx = np.sort(idx)  # sorted fancy-indexing reads the memmap sequentially
        obs = {
            "image": th.as_tensor(np.asarray(images[idx]), device=dev),
            "vector": th.as_tensor(np.asarray(vectors[idx]), device=dev),
        }
        return obs, th.as_tensor(np.asarray(actions[idx]), device=dev, dtype=th.long), th.as_tensor(
            np.asarray(returns[idx]), device=dev
        )

    print(f"[bc] Training: {len(train_idx):,} Train / {n_val:,} Val | {epochs} Epochen | Gerät {dev}")
    model.policy.set_training_mode(True)
    for epoch in range(1, epochs + 1):
        rng.shuffle(train_idx)
        tot_loss = tot_pi = tot_v = 0.0
        n_batches = 0
        for start in range(0, len(train_idx), batch_size):
            obs, acts, rets = batch_tensors(train_idx[start : start + batch_size])
            values, log_prob, entropy = model.policy.evaluate_actions(obs, acts)
            loss_pi = -log_prob.mean()
            loss_v = F.mse_loss(values.flatten(), rets)
            loss = loss_pi + value_coef * loss_v - ent_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            tot_loss += float(loss.detach())
            tot_pi += float(loss_pi.detach())
            tot_v += float(loss_v.detach())
            n_batches += 1

        # validation: NLL + accuracy of the deterministic (argmax) policy
        model.policy.set_training_mode(False)
        with th.no_grad():
            correct = 0
            val_nll = 0.0
            for start in range(0, len(val_idx), 1024):
                obs, acts, _rets = batch_tensors(val_idx[start : start + 1024])
                values, log_prob, _ = model.policy.evaluate_actions(obs, acts)
                val_nll += float(-log_prob.sum())
                dist = model.policy.get_distribution(obs)
                pred = dist.distribution.probs.argmax(dim=1)
                correct += int((pred == acts).sum())
        model.policy.set_training_mode(True)
        print(
            f"[bc] Epoche {epoch}/{epochs}: loss={tot_loss / n_batches:.4f} (pi={tot_pi / n_batches:.4f}, "
            f"v={tot_v / n_batches:.4f}) | val NLL={val_nll / n_val:.4f} | val Trefferquote={correct / n_val:.1%}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"[bc] gespeichert: {out_path}")
    print(f"[bc] nächster Schritt:  python -m ai.v1_0.training.train --config <phase.yaml> --init-from {out_path}")
    return out_path


def evaluate_cloned(cfg: dict, model_path: str | Path, n_episodes: int = 5, seed: int = 123) -> float:
    """Quick sanity rollout of the cloned policy (greedy): mean survived ticks."""
    from ai.v1_0.models.algo import load_trained

    obs_cfg, env_config = make_configs(cfg)
    env = CurveEnv(_stage0_factory(cfg, obs_cfg, seed), config=env_config, seed=seed)
    model = load_trained(model_path, device="cpu")
    ticks: list[int] = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        t = 0
        for _ in range(env.config.max_episode_ticks):
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, _i = env.step(int(np.asarray(action).reshape(-1)[0]))
            t += 1
            if term or trunc:
                break
        ticks.append(t)
    mean_ticks = float(np.mean(ticks))
    print(f"[bc] geklonte Policy (deterministisch): Ø {mean_ticks:.0f} Ticks über {n_episodes} Episoden {ticks}")
    return mean_ticks


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-Cloning-Kickstart vom regelbasierten Bot (siehe Modul-Docstring).")
    parser.add_argument("--config", required=True, help="phaseN.yaml - dieselbe Config wie das spätere PPO-Fine-Tuning")
    parser.add_argument("--out", required=True, help="Zielpfad des BC-Modells (SB3-Zip, für train.py --init-from)")
    parser.add_argument("--data-dir", default=None, help="Datensatz-Verzeichnis (Default: <out-Ordner>/bc_data)")
    parser.add_argument("--transitions", type=int, default=50_000)
    parser.add_argument("--teacher", default="hard", choices=sorted(BOT_DIFFICULTIES) + ["mix"],
                        help="'mix' zieht je Episode zufällig medium/hard/hunter - mehr Zustandsvielfalt im Datensatz")
    parser.add_argument("--explore", type=float, default=0.05, help="Anteil zufällig AUSGEFÜHRTER Aktionen (Labels bleiben Lehrer)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-generate", action="store_true", help="vorhandenen Datensatz in --data-dir wiederverwenden")
    parser.add_argument("--eval-episodes", type=int, default=5, help="0 = Abschluss-Rollout überspringen")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else Path(args.out).parent / "bc_data"
    if not args.skip_generate:
        generate_dataset(cfg, data_dir, n_transitions=args.transitions, teacher=args.teacher, explore_eps=args.explore, seed=args.seed)
    train_bc(
        cfg,
        data_dir,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        value_coef=args.value_coef,
        ent_coef=args.ent_coef,
        seed=args.seed,
    )
    if args.eval_episodes > 0:
        evaluate_cloned(cfg, args.out, n_episodes=args.eval_episodes, seed=args.seed + 1)


if __name__ == "__main__":
    main()
