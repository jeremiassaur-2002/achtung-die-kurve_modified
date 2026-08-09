"""Tests fuer Patch 3: lambda-Renditen, Actor/Critic, Imagination, echter Agent."""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ai.core.config.game_constants import GameConstants
from ai.core.env.engine import CurveEngine, STRAIGHT, TURN_LEFT, TURN_RIGHT
from ai.core.env.observation import ObsConfig
from ai.v1_1.agent import DreamerAgent, DreamerController
from ai.v1_1.models.actor_critic import Actor, ActorCriticConfig, Critic, ReturnNormalizer, lambda_return
from ai.v1_1.models.world_model import WorldModel, WorldModelConfig
from ai.v1_1.training.train_policy import imagine_rollout

IMG = (16, 16, 3)
VEC = 61


def _tiny_wm() -> WorldModel:
    return WorldModel(
        WorldModelConfig(
            image_shape=IMG, vector_dim=VEC, action_dim=3, deter_dim=32,
            stoch_groups=4, stoch_classes=4, hidden_dim=32, cnn_depth=4, vector_embed=16,
        )
    )


def _ac_cfg(**kw) -> ActorCriticConfig:
    base = dict(action_dim=3, hidden_dim=32, imagination_horizon=5)
    base.update(kw)
    return ActorCriticConfig(**base)


# ------------------------------------------------------------ lambda-Renditen


def test_lambda_return_equals_discounted_sum_when_lambda_is_one():
    """lam=1 schaltet das Bootstrapping ab - dann muss exakt die diskontierte
    Belohnungssumme herauskommen (plus der Endwert). Das ist der einzige Fall
    mit geschlossener Loesung und damit der beste Anker gegen Vorzeichen- und
    Indexfehler in der Rueckwaertsschleife."""
    reward = torch.tensor([[1.0, 2.0, 3.0]])
    value = torch.tensor([[0.0, 0.0, 5.0]])
    cont = torch.ones(1, 3)
    gamma = 0.9
    got = lambda_return(reward, value, cont, gamma=gamma, lam=1.0)
    # rueckwaerts: acc_2 = r2 + g*v2 ; acc_1 = r1 + g*acc_2 ; acc_0 = r0 + g*acc_1
    a2 = 3.0 + gamma * 5.0
    a1 = 2.0 + gamma * a2
    a0 = 1.0 + gamma * a1
    assert torch.allclose(got, torch.tensor([[a0, a1, a2]]), atol=1e-5)


def test_lambda_return_stops_at_predicted_death():
    """continue=0 muss die Zukunft abschneiden - sonst rechnet der Critic den
    Ueberlebensbonus ueber den Tod hinaus weiter und der Actor haelt eine
    toedliche Bahn fuer harmlos."""
    reward = torch.tensor([[0.01, 0.01, -1.0]])
    value = torch.tensor([[10.0, 10.0, 10.0]])
    cont = torch.tensor([[1.0, 0.0, 1.0]])
    got = lambda_return(reward, value, cont, gamma=0.99, lam=0.95)
    # Schritt 1 hat continue=0 -> seine Rendite ist genau seine Belohnung
    assert got[0, 1] == pytest.approx(0.01, abs=1e-5)


def test_lambda_return_shape_and_finiteness():
    r = torch.randn(4, 7) * 0.1
    v = torch.randn(4, 7)
    c = torch.rand(4, 7)
    out = lambda_return(r, v, c, 0.997, 0.95)
    assert out.shape == (4, 7) and torch.isfinite(out).all()


# --------------------------------------------------------- Return-Normalizer


def test_return_normalizer_is_floored_at_one():
    """Winzige Renditen duerfen die Vorteile nicht um Groessenordnungen
    aufblasen - genau das wuerde ohne den Deckel bei alive_bonus=0.01 passieren."""
    n = ReturnNormalizer()
    assert n.update(torch.full((100,), 0.01)) == 1.0


def test_return_normalizer_tracks_a_wide_spread():
    n = ReturnNormalizer(decay=0.0)  # decay=0 -> sofort der neue Wert
    scale = n.update(torch.linspace(-10.0, 10.0, 200))
    assert scale > 10.0


# ------------------------------------------------------------- Actor / Critic


def test_actor_starts_uniform():
    """Nullinitialisierte Ausgangsschicht: der Actor darf am Anfang keine
    Vorliebe fuer eine Kurvenrichtung haben, die er sich selbst verstaerkt."""
    actor = Actor(48, _ac_cfg())
    probs = torch.softmax(actor.logits(torch.randn(8, 48)), -1)
    assert torch.allclose(probs, torch.full_like(probs, 1 / 3), atol=1e-5)


def test_actor_unimix_keeps_log_probs_finite():
    """Ohne unimix wird eine Aktionswahrscheinlichkeit exakt 0 und der
    REINFORCE-Term explodiert zu -inf."""
    actor = Actor(48, _ac_cfg(unimix=0.01))
    with torch.no_grad():
        actor.net[-1].weight.fill_(0.0)
        actor.net[-1].bias.copy_(torch.tensor([100.0, -100.0, -100.0]))
    dist = actor.distribution(torch.randn(4, 48))
    lp = dist.log_prob(torch.tensor([1, 2, 1, 2]))
    assert torch.isfinite(lp).all()


def test_critic_slow_target_lags_behind():
    """Das EMA-Ziel muss sich langsamer bewegen als der Critic - sonst ist die
    Rueckkopplung nicht gebrochen und die Wertschaetzung driftet."""
    critic = Critic(48, _ac_cfg(slow_critic_decay=0.98))
    feat = torch.randn(4, 48)
    before = critic.slow_value(feat).clone()
    with torch.no_grad():
        for p in critic.net.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    after_fast = critic.value(feat)
    critic.update_slow()
    after_slow = critic.slow_value(feat)
    assert not torch.allclose(before, after_fast, atol=1e-4)
    assert (after_slow - before).abs().mean() < (after_fast - before).abs().mean()


def test_critic_slow_copy_is_not_optimized():
    critic = Critic(48, _ac_cfg())
    assert all(not p.requires_grad for p in critic.slow.parameters())
    assert all(p.requires_grad for p in critic.net.parameters())


# ---------------------------------------------------------------- Imagination


def _start_state(wm: WorldModel, b: int = 3):
    image = torch.randint(0, 255, (b, 4, *IMG), dtype=torch.uint8)
    vector = torch.randn(b, 4, VEC)
    action = torch.eye(3)[torch.randint(0, 3, (b, 4))]
    with torch.no_grad():
        post, _ = wm.observe(image, vector, action, torch.zeros(b, 4, dtype=torch.bool))
    return post[:, -1]


def test_imagine_rollout_shapes():
    wm = _tiny_wm()
    actor = Actor(wm.feat_dim, _ac_cfg())
    roll = imagine_rollout(wm, actor, _start_state(wm, 3), horizon=5)
    assert roll["feat"].shape == (3, 5, wm.feat_dim)
    for k in ("reward", "continue", "log_prob", "entropy"):
        assert roll[k].shape == (3, 5), k
    assert torch.all((roll["continue"] >= 0) & (roll["continue"] <= 1))


def test_imagination_needs_no_images_and_reaches_the_actor():
    """Der Gradient muss beim Actor ankommen - und NUR dort. Das Weltmodell ist
    in dieser Phase eingefroren; liefe ein Gradient dorthin, koennten Actor und
    Modell gemeinsam in eine bequeme Fantasie abdriften."""
    wm = _tiny_wm()
    for p in wm.parameters():
        p.requires_grad_(False)
    actor = Actor(wm.feat_dim, _ac_cfg())
    roll = imagine_rollout(wm, actor, _start_state(wm, 2), horizon=4)
    (roll["log_prob"].sum() + roll["entropy"].sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
    assert all(p.grad is None for p in wm.parameters())


def test_policy_training_step_reduces_critic_loss():
    """Ein paar echte Trainingsschritte: der Critic muss seine eigenen
    lambda-Renditen besser vorhersagen lernen. Faellt das flach, ist der
    Gradientenpfad irgendwo unterbrochen."""
    torch.manual_seed(0)
    wm = _tiny_wm()
    for p in wm.parameters():
        p.requires_grad_(False)
    cfg = _ac_cfg(imagination_horizon=5)
    actor, critic = Actor(wm.feat_dim, cfg), Critic(wm.feat_dim, cfg)
    opt = torch.optim.Adam(critic.net.parameters(), lr=3e-3)
    start = _start_state(wm, 8)
    losses = []
    for _ in range(40):
        roll = imagine_rollout(wm, actor, start, horizon=5)
        with torch.no_grad():
            returns = lambda_return(roll["reward"], critic.slow_value(roll["feat"]), roll["continue"], 0.99, 0.95)
        loss = critic.twohot.loss(critic.logits(roll["feat"].detach()), returns).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        critic.update_slow()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], f"Critic-Verlust fiel nicht: {losses[0]:.3f} -> {losses[-1]:.3f}"


# --------------------------------------------------------------------- Agent


def _agent() -> DreamerAgent:
    wm = _tiny_wm()
    actor = Actor(wm.feat_dim, _ac_cfg())
    obs_cfg = ObsConfig(obs_resolution=IMG[0], frame_stack=1, n_rays=36, ray_range_px=64.0)
    return DreamerAgent(wm, actor, obs_cfg)


def _engine(seed: int = 0, ticks: int = 0) -> CurveEngine:
    eng = CurveEngine(GameConstants(256))
    eng.reset(["fred", "greenlee"], seed=seed)
    rng = random.Random(seed)
    for _ in range(ticks):
        if not eng.players["fred"].alive:
            break
        eng.step({n: rng.choice([0, 1, 2]) for n in eng.players})
    return eng


def test_agent_returns_valid_actions_in_the_real_engine():
    agent = _agent()
    eng = _engine(seed=1)
    agent.reset()
    for _ in range(12):
        a = agent.act(eng, "fred")
        assert a in (TURN_LEFT, STRAIGHT, TURN_RIGHT)
        eng.step({"fred": a, "greenlee": STRAIGHT})


def test_agent_reset_clears_the_recurrent_state():
    """Ein uebriggebliebener Zustand aus der letzten Runde aeussert sich als
    'der Agent spielt am Anfang komisch' und ist schwer zu finden - deshalb
    hier explizit geprueft."""
    agent = _agent()
    eng = _engine(seed=2, ticks=25)
    if not eng.players["fred"].alive:
        pytest.skip("Held bereits tot")
    agent.reset()
    for _ in range(5):
        agent.act(eng, "fred")
    assert agent._state is not None and agent._prev_action is not None
    agent.reset()
    assert agent._state is None and agent._prev_action is None


def test_agent_is_deterministic_without_sampling():
    agent = _agent()
    eng = _engine(seed=3, ticks=10)
    if not eng.players["fred"].alive:
        pytest.skip("Held bereits tot")
    agent.reset()
    first = agent.act(eng, "fred", sample=False)
    agent.reset()
    second = agent.act(eng, "fred", sample=False)
    assert first == second


def test_agent_state_actually_evolves():
    """Der RSSM-Zustand muss sich ueber die Ticks fortschreiben - bliebe er
    konstant, waere der Agent effektiv gedaechtnislos und der ganze rekurrente
    Aufbau sinnlos."""
    agent = _agent()
    eng = _engine(seed=4)
    agent.reset()
    agent.act(eng, "fred")
    h1 = agent._state.deter.clone()
    for _ in range(3):
        eng.step({"fred": STRAIGHT, "greenlee": STRAIGHT})
        agent.act(eng, "fred")
    assert not torch.allclose(h1, agent._state.deter, atol=1e-5)


def test_dreamer_controller_interface():
    ctrl = DreamerController(_agent())
    eng = _engine(seed=5, ticks=8)
    if not eng.players["fred"].alive:
        pytest.skip("Held bereits tot")
    ctrl.reset("fred")
    assert ctrl.act(eng, "fred", None) in (TURN_LEFT, STRAIGHT, TURN_RIGHT)


def test_agent_rejects_frame_stacking():
    """v1_1 arbeitet ohne Frame-Stack; ein falsch konfigurierter Stack wuerde
    sonst still eine andere Kanalzahl in den Encoder schieben."""
    wm = _tiny_wm()
    actor = Actor(wm.feat_dim, _ac_cfg())
    obs_cfg = ObsConfig(obs_resolution=IMG[0], frame_stack=4, n_rays=36, ray_range_px=64.0)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        wm_path, pol_path = Path(d) / "wm.pt", Path(d) / "p.pt"
        torch.save({"model": wm.state_dict(), "cfg": wm.cfg.__dict__}, wm_path)
        torch.save({"actor": actor.state_dict(), "ac_cfg": actor.cfg.__dict__}, pol_path)
        with pytest.raises(ValueError, match="frame_stack"):
            DreamerAgent.load(pol_path, wm_path, obs_cfg)
