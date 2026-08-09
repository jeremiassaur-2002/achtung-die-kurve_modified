"""Tests fuer den Dreamer-Kern: Replay, symlog/twohot, RSSM, Weltmodell.

Alle Modelle hier sind absichtlich winzig (deter 32, 4x4 Latents, cnn_depth 4).
Geprueft wird Korrektheit, nicht Lernleistung - dafuer reicht ein Modell, das in
Sekunden auf einer CPU laeuft.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ai.v1_1.data.replay import ShardReplay
from ai.v1_1.models.nets import TwoHotEncoding, symexp, symlog
from ai.v1_1.models.rssm import RSSM
from ai.v1_1.models.world_model import WorldModel, WorldModelConfig

IMG = (16, 16, 3)
VEC = 61


def _tiny_cfg(**kw) -> WorldModelConfig:
    base = dict(
        image_shape=IMG,
        vector_dim=VEC,
        action_dim=3,
        deter_dim=32,
        stoch_groups=4,
        stoch_classes=4,
        hidden_dim=32,
        cnn_depth=4,
        vector_embed=16,
    )
    base.update(kw)
    return WorldModelConfig(**base)


def _fake_dataset(tmp_path, episodes=4, lengths=(30, 5, 60, 12)):
    """Shards im Format von data/collect.py schreiben - inklusive einer Episode,
    die kuerzer ist als die spaeter gesampelte Sequenzlaenge."""
    shards = tmp_path / "shards"
    shards.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(episodes):
        n = lengths[i % len(lengths)]
        dones = np.zeros(n, bool)
        dones[-1] = True
        np.savez_compressed(
            shards / f"ep_w0_{i:05d}.npz",
            frames=rng.integers(0, 255, (n, *IMG), dtype=np.uint8),
            vectors=rng.standard_normal((n, VEC)).astype(np.float32),
            actions=rng.integers(0, 3, n).astype(np.int8),
            rewards=np.full(n, 0.01, np.float32),
            dones=dones,
            planner_survived=np.full(n, 45, np.int16),
        )
    return tmp_path


def _batch(b=2, seq_len=6, device="cpu"):
    return dict(
        image=torch.randint(0, 255, (b, seq_len, *IMG), dtype=torch.uint8, device=device),
        vector=torch.randn(b, seq_len, VEC, device=device),
        action=torch.eye(3, device=device)[torch.randint(0, 3, (b, seq_len))],
        reward=torch.randn(b, seq_len, device=device) * 0.1,
        is_first=torch.zeros(b, seq_len, dtype=torch.bool, device=device),
        is_terminal=torch.zeros(b, seq_len, dtype=torch.bool, device=device),
        mask=torch.ones(b, seq_len, device=device),
        ticks_to_death=torch.rand(b, seq_len, device=device) * 120.0,
    )


# ---------------------------------------------------------------- symlog/twohot


def test_symlog_roundtrip_over_many_magnitudes():
    x = torch.tensor([-1e4, -100.0, -1.0, -0.01, 0.0, 0.01, 1.0, 100.0, 1e4])
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-4, atol=1e-3)


def test_twohot_is_a_distribution_and_inverts():
    th = TwoHotEncoding(bins=255)
    vals = torch.tensor([-1.0, -0.5, 0.0, 0.01, 0.5, 2.0, 50.0])
    enc = th.encode(vals)
    assert torch.allclose(enc.sum(-1), torch.ones_like(vals), atol=1e-5)
    assert (enc > 0).sum(-1).max() <= 2, "twohot darf hoechstens zwei Stuetzstellen belegen"
    dec = th.decode(torch.log(enc.clamp(min=1e-9)))
    assert torch.allclose(dec, vals, rtol=1e-3, atol=1e-3)


def test_twohot_loss_is_minimal_at_the_target():
    """Der Verlust muss am richtigen Wert kleiner sein als daneben - sonst
    lernt der Belohnungskopf systematisch falsch."""
    th = TwoHotEncoding(bins=255)
    target = torch.tensor([0.5])
    good = torch.log(th.encode(torch.tensor([0.5])).clamp(min=1e-9))
    bad = torch.log(th.encode(torch.tensor([-1.0])).clamp(min=1e-9))
    assert float(th.loss(good, target)) < float(th.loss(bad, target))


# --------------------------------------------------------------------- Replay


def test_replay_reads_shards_and_reports_stats(tmp_path):
    r = ShardReplay(_fake_dataset(tmp_path))
    s = r.stats()
    assert s["episodes"] == 4 and s["total_ticks"] == 30 + 5 + 60 + 12
    assert r.shapes() == {"image": IMG, "vector": (VEC,), "action": (3,)}


def test_replay_batch_shapes_and_onehot(tmp_path):
    r = ShardReplay(_fake_dataset(tmp_path), seed=1)
    b = r.sample(5, 16)
    assert b.image.shape == (5, 16, *IMG)
    assert b.vector.shape == (5, 16, VEC) and b.action.shape == (5, 16, 3)
    on = b.action.sum(-1)
    assert np.all((on == 1.0) | (on == 0.0)), "Aktionen muessen one-hot (oder Padding) sein"


def test_replay_pads_short_episodes_instead_of_dropping_them(tmp_path):
    """Kurze Episoden sind die toedlichen - sie duerfen nicht wegfallen, sonst
    lernt das Modell nie, wie ein Aufprall aussieht."""
    r = ShardReplay(_fake_dataset(tmp_path, episodes=1, lengths=(5,)), seed=2)
    b = r.sample(3, 16)
    assert b.mask.shape == (3, 16)
    assert np.all(b.mask[:, :5] == 1.0) and np.all(b.mask[:, 5:] == 0.0)
    assert np.all(b.action[:, 5:] == 0.0)


def test_replay_never_crosses_episode_boundaries(tmp_path):
    """Hoechstens ein Terminal pro Sequenz, und nur am Ende des gueltigen Teils -
    saehe das Modell zwei Tode in einer Sequenz, waere ueber eine Episodengrenze
    hinweg gesampelt worden."""
    r = ShardReplay(_fake_dataset(tmp_path), seed=3)
    b = r.sample(24, 20)
    for i in range(len(b)):
        valid = b.is_terminal[i][b.mask[i] > 0]
        assert valid.sum() <= 1
        if valid.sum() == 1:
            assert valid[-1], "ein Terminal darf nur am Ende des gueltigen Teils stehen"


def test_replay_raises_helpfully_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="collect"):
        ShardReplay(tmp_path)


# ----------------------------------------------------------------------- RSSM


def test_rssm_observe_shapes_and_feature_dim():
    rssm = RSSM(embed_dim=16, action_dim=3, deter_dim=32, stoch_groups=4, stoch_classes=4, hidden_dim=32)
    b, seq_len = 3, 7
    post, prior = rssm.observe(
        torch.randn(b, seq_len, 16), torch.eye(3)[torch.randint(0, 3, (b, seq_len))], torch.zeros(b, seq_len, dtype=torch.bool)
    )
    assert post.deter.shape == (b, seq_len, 32)
    assert post.stoch.shape == (b, seq_len, 4, 4)
    assert post.feature.shape == (b, seq_len, 32 + 16)
    assert prior.deter.shape == post.deter.shape


def test_rssm_stoch_is_onehot_per_group():
    """Straight-Through muss vorwaerts eine harte One-Hot liefern - sonst waere
    der Latent doch wieder kontinuierlich."""
    rssm = RSSM(embed_dim=8, action_dim=3, deter_dim=16, stoch_groups=4, stoch_classes=6, hidden_dim=16)
    post, _ = rssm.observe(torch.randn(2, 3, 8), torch.eye(3)[torch.randint(0, 3, (2, 3))], torch.zeros(2, 3, dtype=torch.bool))
    assert torch.allclose(post.stoch.sum(-1), torch.ones(2, 3, 4), atol=1e-5)


def test_rssm_is_first_resets_the_state():
    """Am Episodenanfang darf nichts aus der vorigen Episode durchsickern.

    Verglichen wird ueber zwei Laeufe mit unterschiedlicher Vorgeschichte: OHNE
    Reset muss Schritt 2 die Aenderung noch spueren, MIT Reset nicht mehr.
    Genuegend Gruppen/Klassen, damit die argmax-Stichprobe im eval-Modus nicht
    zufaellig auf dieselbe One-Hot faellt und den Test blind macht.
    """
    rssm = RSSM(embed_dim=8, action_dim=3, deter_dim=16, stoch_groups=8, stoch_classes=8, hidden_dim=16).eval()
    embed = torch.randn(1, 4, 8)
    embed2 = embed.clone()
    embed2[0, :2] = torch.randn(2, 8)
    action = torch.eye(3)[torch.tensor([[0, 1, 2, 0]])]

    # Der Posterior sieht die Beobachtung direkt - hier muss sich sofort etwas
    # aendern, sonst wuerde der Test unten nur Rauschen messen.
    no_reset = torch.zeros(1, 4, dtype=torch.bool)
    a, _ = rssm.observe(embed, action, no_reset)
    b, _ = rssm.observe(embed2, action, no_reset)
    assert not torch.allclose(a.logits[0, 0], b.logits[0, 0], atol=1e-5)
    assert not torch.allclose(a.deter[0, 2], b.deter[0, 2], atol=1e-5), "ohne Reset muesste Schritt 2 die Vorgeschichte spueren"

    reset = no_reset.clone()
    reset[0, 2] = True
    a2, _ = rssm.observe(embed, action, reset)
    b2, _ = rssm.observe(embed2, action, reset)
    assert torch.allclose(a2.deter[0, 2], b2.deter[0, 2], atol=1e-5), "is_first muss den Zustand nullen"


def test_rssm_free_nats_floor_the_kl():
    """Unter den free nats darf der KL-Verlust nicht weiter sinken."""
    rssm = RSSM(embed_dim=8, action_dim=3, deter_dim=16, stoch_groups=2, stoch_classes=4, hidden_dim=16)
    post, prior = rssm.observe(torch.randn(2, 3, 8), torch.eye(3)[torch.randint(0, 3, (2, 3))], torch.zeros(2, 3, dtype=torch.bool))
    # identische Verteilungen -> roher KL = 0, geclampter Verlust = beta-Summe
    same = post
    kl, dyn, rep = rssm.kl_loss(same, same, free_nats=1.0, beta_dyn=0.5, beta_rep=0.1)
    assert torch.allclose(dyn, torch.zeros_like(dyn), atol=1e-4)
    assert torch.allclose(kl, torch.full_like(kl, 0.6), atol=1e-4)


def test_rssm_imagine_needs_no_observation():
    rssm = RSSM(embed_dim=8, action_dim=3, deter_dim=16, stoch_groups=2, stoch_classes=4, hidden_dim=16)
    state = rssm.initial(4, torch.device("cpu"))
    dreamed = rssm.imagine(state, torch.eye(3)[torch.randint(0, 3, (4, 9))])
    assert dreamed.deter.shape == (4, 9, 16)


# ---------------------------------------------------------------- Weltmodell


def test_world_model_loss_runs_and_all_params_get_gradients():
    wm = WorldModel(_tiny_cfg())
    loss, post = wm.loss(**_batch())
    loss.total.backward()
    missing = [n for n, p in wm.named_parameters() if p.grad is None]
    assert not missing, f"kein Gradient bei: {missing}"
    for key in ("image", "vector", "reward", "continue", "kl", "reward_mae", "continue_acc"):
        assert key in loss.parts


def test_world_model_mask_excludes_padding():
    """Padding-Schritte duerfen den Verlust nicht beeinflussen - sonst lernt das
    Modell, schwarze Platzhalterbilder zu rekonstruieren."""
    torch.manual_seed(0)
    wm = WorldModel(_tiny_cfg()).eval()
    b = _batch(b=1, seq_len=6)
    b["mask"][:, 3:] = 0.0
    with torch.no_grad():
        base, _ = wm.loss(**b)
        b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        b2["image"][:, 3:] = 0  # Padding aendern
        b2["vector"][:, 3:] = 99.0
        changed, _ = wm.loss(**b2)
    assert abs(float(base.total) - float(changed.total)) < 1e-3


def test_world_model_overfits_a_single_batch():
    """Der schaerfste Korrektheitstest: auf EINEM festen Batch muss der Verlust
    deutlich fallen. Tut er das nicht, ist irgendwo der Gradientenpfad
    unterbrochen - und das faellt sonst erst nach Stunden GPU-Zeit auf."""
    torch.manual_seed(0)
    wm = WorldModel(_tiny_cfg())
    batch = _batch(b=2, seq_len=6)
    # Strukturierte statt verrauschte Bilder: reines Pixelrauschen ist prinzipiell
    # nicht komprimierbar, ein Misserfolg darauf saegte nichts ueber den
    # Gradientenpfad aus - und genau der soll hier geprueft werden.
    for i in range(batch["image"].shape[0]):
        for t_ in range(batch["image"].shape[1]):
            batch["image"][i, t_] = (40 * (i + 1) + 30 * t_) % 255
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    with torch.no_grad():
        first = float(wm.loss(**batch)[0].total)
    for _ in range(120):
        loss, _ = wm.loss(**batch)
        opt.zero_grad(set_to_none=True)
        loss.total.backward()
        opt.step()
    with torch.no_grad():
        last = float(wm.loss(**batch)[0].total)
    assert last < 0.6 * first, f"Verlust fiel nur von {first:.1f} auf {last:.1f}"


def test_world_model_dream_shapes_and_no_engine_needed():
    wm = WorldModel(_tiny_cfg()).eval()
    b = _batch(b=2, seq_len=4)
    post, _ = wm.observe(b["image"], b["vector"], b["action"], b["is_first"])
    dreamed = wm.dream(post[:, -1], torch.eye(3)[torch.randint(0, 3, (2, 5))])
    assert dreamed["image"].shape == (2, 5, *IMG)
    assert dreamed["vector"].shape == (2, 5, VEC)
    assert dreamed["reward"].shape == (2, 5) and dreamed["continue"].shape == (2, 5)
    assert torch.all((dreamed["continue"] >= 0) & (dreamed["continue"] <= 1))


def test_world_model_rejects_non_square_images():
    with pytest.raises(ValueError):
        WorldModel(_tiny_cfg(image_shape=(16, 32, 3)))


def test_dream_diagnostic_end_to_end(tmp_path):
    from ai.v1_1.training.train_world_model import dream_diagnostic

    r = ShardReplay(_fake_dataset(tmp_path, episodes=3, lengths=(40, 40, 40)), seed=5)
    wm = WorldModel(_tiny_cfg())
    d = dream_diagnostic(wm, r, torch.device("cpu"), context=4, horizon=6)
    assert set(d) == {"dream_vec_mae", "dream_img_mae", "dream_horizon"}
    assert np.isfinite(d["dream_vec_mae"]) and d["dream_horizon"] == 6
