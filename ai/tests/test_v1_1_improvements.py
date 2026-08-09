"""Tests fuer Patch 4 - die vier Korrekturen am Lernaufbau.

Jede hier gepruefte Eigenschaft entspricht einem Fehler, der ohne Test
UNBEMERKT bliebe, weil das Training in allen vier Faellen weiterhin sauber
durchlaeuft und plausible Zahlen ausgibt.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ai.v1_1.data.replay import ShardReplay
from ai.v1_1.models.actor_critic import Actor, ActorCriticConfig
from ai.v1_1.models.world_model import WorldModel, WorldModelConfig

IMG = (16, 16, 3)
VEC = 61


def _dataset(tmp_path, fatal=(True, True, False), lengths=(40, 25, 60)):
    """`fatal=False` = Episode endet per Zeitlimit statt Tod."""
    shards = tmp_path / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i, (is_fatal, n) in enumerate(zip(fatal, lengths)):
        dones = np.zeros(n, bool)
        dones[-1] = is_fatal
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


def _tiny_wm(**kw) -> WorldModel:
    base = dict(
        image_shape=IMG, vector_dim=VEC, action_dim=3, deter_dim=32,
        stoch_groups=4, stoch_classes=4, hidden_dim=32, cnn_depth=4, vector_embed=16,
    )
    base.update(kw)
    return WorldModel(WorldModelConfig(**base))


# ------------------------------------------------- 1) Restlebensdauer (dicht)


def test_ticks_to_death_counts_down_to_zero_at_a_real_death():
    dones = np.zeros(10, bool)
    dones[-1] = True
    ttd = ShardReplay._ticks_to_death(dones, cap=120.0)
    assert ttd.tolist() == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


def test_ticks_to_death_is_capped():
    """Weit vom Tod entfernt ist die exakte Restdauer weder vorhersagbar noch
    interessant - relevant ist nur die Naehe."""
    dones = np.zeros(200, bool)
    dones[-1] = True
    ttd = ShardReplay._ticks_to_death(dones, cap=50.0)
    assert ttd.max() == 50.0 and ttd[-1] == 0.0


def test_ticks_to_death_invents_no_death_on_a_time_limit():
    """Eine per Zeitlimit abgeschnittene Episode ist KEIN Tod. Wuerde hier
    runtergezaehlt, lernte das Modell, dass die Zeit an sich toetet."""
    ttd = ShardReplay._ticks_to_death(np.zeros(30, bool), cap=120.0)
    assert np.all(ttd == 120.0)


def test_replay_provides_ticks_to_death(tmp_path):
    r = ShardReplay(_dataset(tmp_path), seed=0, ttd_cap=60.0)
    b = r.sample(6, 20)
    assert b.ticks_to_death.shape == (6, 20)
    assert b.ticks_to_death.min() >= 0.0 and b.ticks_to_death.max() <= 60.0


def test_survival_head_gets_gradients_and_predicts_the_target():
    """Der Kopf muss lernbar sein - und zwar auf einem konstanten Ziel in
    wenigen Schritten, sonst stimmt etwas mit der twohot-Kodierung nicht."""
    torch.manual_seed(0)
    wm = _tiny_wm()
    b, seq_len = 2, 4
    batch = dict(
        image=torch.randint(0, 255, (b, seq_len, *IMG), dtype=torch.uint8),
        vector=torch.randn(b, seq_len, VEC),
        action=torch.eye(3)[torch.randint(0, 3, (b, seq_len))],
        reward=torch.zeros(b, seq_len),
        is_first=torch.zeros(b, seq_len, dtype=torch.bool),
        is_terminal=torch.zeros(b, seq_len, dtype=torch.bool),
        mask=torch.ones(b, seq_len),
        ticks_to_death=torch.full((b, seq_len), 7.0),
    )
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    first = wm.loss(**batch)[0].parts["survival_mae"]
    for _ in range(80):
        loss, _ = wm.loss(**batch)
        opt.zero_grad(set_to_none=True)
        loss.total.backward()
        opt.step()
    assert all(p.grad is not None for p in wm.survival_head.parameters())
    assert wm.loss(**batch)[0].parts["survival_mae"] < first


def test_world_model_still_works_without_ticks_to_death():
    """Rueckwaertskompatibel: ein alter Aufruf ohne das neue Ziel darf nicht
    brechen, der Survival-Term faellt dann einfach weg."""
    wm = _tiny_wm()
    b, seq_len = 2, 3
    loss, _ = wm.loss(
        image=torch.randint(0, 255, (b, seq_len, *IMG), dtype=torch.uint8),
        vector=torch.randn(b, seq_len, VEC),
        action=torch.eye(3)[torch.randint(0, 3, (b, seq_len))],
        reward=torch.zeros(b, seq_len),
        is_first=torch.zeros(b, seq_len, dtype=torch.bool),
        is_terminal=torch.zeros(b, seq_len, dtype=torch.bool),
        mask=torch.ones(b, seq_len),
    )
    assert loss.parts["survival"] == 0.0 and torch.isfinite(loss.total)


# --------------------------------------------- 2) Terminal-Ungleichgewicht


def test_terminal_fraction_forces_deaths_into_the_batch(tmp_path):
    """Der Kern der Sache: ohne Quote enthaelt nur etwa jedes achte Fenster
    einen Tod, mit Quote der geforderte Anteil."""
    r = ShardReplay(_dataset(tmp_path, fatal=(True, True, False), lengths=(200, 200, 200)), seed=0, terminal_fraction=0.5)
    b = r.sample(8, 16)
    with_death = int(((b.is_terminal) & (b.mask > 0)).any(axis=1).sum())
    assert with_death >= 4, f"nur {with_death}/8 Sequenzen enthalten einen Tod"


def test_terminal_fraction_zero_keeps_old_behaviour(tmp_path):
    r = ShardReplay(_dataset(tmp_path, lengths=(300, 300, 300)), seed=0, terminal_fraction=0.0)
    b = r.sample(8, 8)
    assert int((b.is_terminal & (b.mask > 0)).any(axis=1).sum()) <= 2


def test_replay_knows_which_episodes_end_fatally(tmp_path):
    r = ShardReplay(_dataset(tmp_path, fatal=(True, False, True)), seed=0)
    assert len(r._fatal) == 2
    assert r.stats()["fatal_episodes"] == 2


def test_continue_pos_weight_makes_terminals_count():
    """Ohne Gewichtung ist der Verlust eines uebersehenen Todes genauso gross
    wie der eines uebersehenen normalen Schritts - bei 1:1000 Verhaeltnis
    optimiert das Modell den Tod schlicht weg."""
    b, seq_len = 1, 4
    common = dict(
        image=torch.zeros(b, seq_len, *IMG, dtype=torch.uint8),
        vector=torch.zeros(b, seq_len, VEC),
        action=torch.eye(3)[torch.zeros(b, seq_len, dtype=torch.long)],
        reward=torch.zeros(b, seq_len),
        is_first=torch.zeros(b, seq_len, dtype=torch.bool),
        mask=torch.ones(b, seq_len),
    )
    terminal = torch.zeros(b, seq_len, dtype=torch.bool)
    terminal[0, -1] = True

    torch.manual_seed(0)
    weighted = _tiny_wm(continue_pos_weight=50.0)
    torch.manual_seed(0)
    plain = _tiny_wm(continue_pos_weight=1.0)
    with torch.no_grad():
        lw = weighted.loss(**common, is_terminal=terminal)[0].parts["continue"]
        lp = plain.loss(**common, is_terminal=terminal)[0].parts["continue"]
    assert lw > lp, "der gewichtete Verlust muss den seltenen Tod staerker bestrafen"


def test_terminal_recall_is_reported_separately():
    """Die Gesamt-Trefferquote liegt wegen der Seltenheit ohnehin ueber 99% und
    sagt nichts - deshalb eine eigene Kennzahl nur auf den Terminals."""
    wm = _tiny_wm()
    b, seq_len = 2, 4
    terminal = torch.zeros(b, seq_len, dtype=torch.bool)
    terminal[:, -1] = True
    loss, _ = wm.loss(
        image=torch.zeros(b, seq_len, *IMG, dtype=torch.uint8),
        vector=torch.zeros(b, seq_len, VEC),
        action=torch.eye(3)[torch.zeros(b, seq_len, dtype=torch.long)],
        reward=torch.zeros(b, seq_len),
        is_first=torch.zeros(b, seq_len, dtype=torch.bool),
        is_terminal=terminal,
        mask=torch.ones(b, seq_len),
    )
    assert 0.0 <= loss.parts["terminal_recall"] <= 1.0


# -------------------------------------------------- 3) Train/Val-Trennung


def test_split_separates_whole_episodes(tmp_path):
    """Ueber Schritte zu splitten waere wertlos: zwei Fenster derselben Episode
    ueberlappen stark, die Validierung wuerde nur Auswendiglernen messen."""
    d = _dataset(tmp_path, fatal=(True,) * 10, lengths=(50,) * 10)
    train = ShardReplay(d, split="train", val_fraction=0.2)
    val = ShardReplay(d, split="val", val_fraction=0.2)
    assert set(train.shards).isdisjoint(set(val.shards))
    assert len(train.shards) + len(val.shards) == 10
    assert len(val.shards) == 2


def test_split_rejects_nonsense(tmp_path):
    with pytest.raises(ValueError):
        ShardReplay(_dataset(tmp_path), split="testing")


# ------------------------------------------ 4) KL-Anker gegen Abdriften


def test_bc_kl_is_zero_against_an_identical_policy():
    from ai.v1_1.models.actor_critic import Actor as A

    cfg = ActorCriticConfig(action_dim=3, hidden_dim=32)
    actor = A(48, cfg)
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    import copy

    ref = copy.deepcopy(actor)
    feat = torch.randn(16, 48)
    cur = torch.log_softmax(actor.logits(feat), -1)
    with torch.no_grad():
        ref_p = torch.softmax(ref.logits(feat), -1)
    kl = (ref_p * (torch.log(ref_p.clamp(min=1e-8)) - cur)).sum(-1).mean()
    assert float(kl.detach()) == pytest.approx(0.0, abs=1e-5)


def test_bc_kl_grows_as_the_policy_drifts():
    """Der Anker muss anziehen, je weiter der Actor sich entfernt - sonst
    begrenzt er das Abdriften in Weltmodell-Luecken nicht."""
    cfg = ActorCriticConfig(action_dim=3, hidden_dim=32)
    actor = Actor(48, cfg)
    import copy

    ref = copy.deepcopy(actor)
    feat = torch.randn(32, 48)

    def kl_now() -> float:
        cur = torch.log_softmax(actor.logits(feat), -1)
        with torch.no_grad():
            ref_p = torch.softmax(ref.logits(feat), -1)
        return float((ref_p * (torch.log(ref_p.clamp(min=1e-8)) - cur)).sum(-1).mean())

    small = None
    with torch.no_grad():
        actor.net[-1].bias.add_(torch.tensor([0.5, -0.5, 0.0]))
        small = kl_now()
        actor.net[-1].bias.add_(torch.tensor([3.0, -3.0, 0.0]))
        large = kl_now()
    assert 0.0 < small < large


def test_dream_reports_ticks_to_death():
    """Die getraeumte Restlebensdauer ist die Diagnose waehrend des
    Policy-Trainings: faellt sie waehrend die Rendite steigt, nutzt der Actor
    ein Leck im Weltmodell aus."""
    wm = _tiny_wm().eval()
    b = 2
    image = torch.randint(0, 255, (b, 3, *IMG), dtype=torch.uint8)
    vector = torch.randn(b, 3, VEC)
    action = torch.eye(3)[torch.randint(0, 3, (b, 3))]
    post, _ = wm.observe(image, vector, action, torch.zeros(b, 3, dtype=torch.bool))
    dreamed = wm.dream(post[:, -1], torch.eye(3)[torch.randint(0, 3, (b, 4))])
    assert dreamed["ticks_to_death"].shape == (b, 4)
    assert torch.isfinite(dreamed["ticks_to_death"]).all()
