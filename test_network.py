"""EntityPolicyNet / PolicyValueNet: masking, shapes, slot wiring (permutation tests), checkpoints."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from cardgame.actions import ActionSpace
from cardgame.cards import load_ruleset
from cardgame.engine import Game
from cardgame.features import GLOBAL_FEATURES, ObservationEncoder
from cardgame.rl.agent import CheckpointError, PPOAgent, net_from_checkpoint
from cardgame.rl.network import EntityPolicyNet, PolicyValueNet, build_net

CONFIG = load_ruleset()
ENC = ObservationEncoder(CONFIG)
SP = ActionSpace(CONFIG.max_hand_size, CONFIG.zone_capacity)
H, Z = CONFIG.max_hand_size, CONFIG.zone_capacity


def batch(n_games=15):
    g, rng, X, M = Game(CONFIG), random.Random(0), [], []
    for s in range(n_games):
        g.reset(s)
        while not g.done:
            m = g.legal_mask()
            X.append(ENC.encode(g.observe(g.current_player()), m))
            M.append(m)
            la = g.legal_actions()
            g.step(la[rng.randrange(len(la))])
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(M))


X, M = batch()
NETS = {
    "entity": lambda: EntityPolicyNet(ENC.layout(), d_model=32, ctx_dim=64, pair_dim=32),
    "entity_shared_attn": lambda: EntityPolicyNet(ENC.layout(), d_model=32, ctx_dim=64, pair_dim=32,
                                                  attention_layers=1, shared_trunk=True),
    "mlp": lambda: PolicyValueNet(ENC.dim, SP.n, (64, 64)),
}


@pytest.mark.parametrize("kind", NETS)
def test_masked_sampling_shapes_and_gradients(kind):
    torch.manual_seed(0)
    net = NETS[kind]()
    for _ in range(5):
        a, logp = net.act(X, M)
        assert M[torch.arange(len(a)), a].all() and torch.isfinite(logp).all()
    lp, ent, v = net.evaluate(X, M, a)
    assert torch.allclose(lp, logp, atol=1e-5)
    assert lp.shape == ent.shape == v.shape == (len(X),)
    assert torch.isfinite(ent).all() and (ent >= -1e-6).all()
    (-(lp.mean()) + v.pow(2).mean() - ent.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    d, _ = net.act(X, M, deterministic=True)
    assert M[torch.arange(len(d)), d].all()


@pytest.mark.parametrize("kind", NETS)
def test_spec_round_trip(kind):
    net = NETS[kind]()
    clone = build_net(net.spec())
    clone.load_state_dict(net.state_dict())
    assert torch.allclose(clone.masked_logits(X[:20], M[:20]), net.masked_logits(X[:20], M[:20]))
    assert torch.allclose(clone.get_value(X[:20]), net.get_value(X[:20]))


def test_shared_trunk_shares_parameters():
    net = NETS["entity_shared_attn"]()
    assert net.value_tower is net.policy_tower
    assert NETS["entity"]().value_tower is not NETS["entity"]().policy_tower


def test_empty_zones_give_finite_outputs():
    x = torch.zeros(3, ENC.dim)
    mask = torch.zeros(3, SP.n, dtype=torch.bool)
    mask[:, 0] = True
    for kind in NETS:
        logits, value = NETS[kind]().forward(x, mask)
        assert torch.isfinite(value).all() and torch.isfinite(logits[:, 0]).all()


def _swap_slots(x: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Swap two entity slots (features, ids, presence and their attack-preview rows/columns)."""
    y = x.clone()
    G, F = ENC.G, ENC.F
    for off, width in ((G, F), (ENC.off_ids, 1), (ENC.off_mask, 1)):
        a, b = off + i * width, off + j * width
        y[:, a:a + width], y[:, b:b + width] = x[:, b:b + width], x[:, a:a + width]
    n_a, n_t, P = 2 * Z, 2 * Z + 1, ENC.P
    prev = y[:, ENC.off_preview:ENC.off_ids].reshape(-1, n_a, n_t, P).clone()

    def roles(e):  # (attacker slot, target slot) of an entity slot, None where it has no role
        if H <= e < H + Z:
            return e - H, None
        if H + Z <= e < H + 2 * Z:
            return e - H, e - H
        if e >= H + 2 * Z:
            return None, e - H - 2 * Z
        return None, None

    (ai, ti), (aj, tj) = roles(i), roles(j)
    if ai is not None and aj is not None:
        prev[:, [ai, aj]] = prev[:, [aj, ai]]
    if ti is not None and tj is not None:
        prev[:, :, [ti, tj]] = prev[:, :, [tj, ti]]
    y[:, ENC.off_preview:ENC.off_ids] = prev.reshape(len(y), -1)
    return y


def _rich_rows():
    """Rows with >= 3 hand cards, 2 own backline units and 2 enemy backline units."""
    p = ENC.split(X)
    present = p.mask
    ok = (present[:, :3].all(1) & present[:, H:H + 2].all(1) & present[:, H + 2 * Z:H + 2 * Z + 2].all(1))
    rows = X[ok][:8]
    assert len(rows) >= 2
    return rows


def _perturbed_entity_net():
    torch.manual_seed(0)
    net = NETS["entity"]()
    with torch.no_grad():
        for p in net.parameters():  # make the heads non-trivial
            p.add_(torch.randn_like(p) * 0.05)
    return net


def test_slot_wiring_is_equivariant():
    net = _perturbed_entity_net()
    x = _rich_rows()
    base = net.policy_logits(x)
    att = lambda logits: logits[:, SP.ATTACK0:].reshape(-1, 2 * Z, 2 * Z + 1)  # noqa: E731
    # hand slots 0 <-> 1: PLAY(0) and PLAY(1) swap, nothing else changes
    sw = net.policy_logits(_swap_slots(x, 0, 1))
    assert torch.allclose(sw[:, SP.PLAY0], base[:, SP.PLAY0 + 1], atol=1e-5)
    assert torch.allclose(sw[:, SP.PLAY0 + 1], base[:, SP.PLAY0], atol=1e-5)
    assert torch.allclose(sw[:, SP.MOVE0:], base[:, SP.MOVE0:], atol=1e-5)
    # my backline 0 <-> 1: MOVE(0)/MOVE(1) and attacker rows 0/1 swap
    sw = net.policy_logits(_swap_slots(x, H, H + 1))
    assert torch.allclose(sw[:, SP.MOVE0], base[:, SP.MOVE0 + 1], atol=1e-5)
    assert torch.allclose(att(sw)[:, 0], att(base)[:, 1], atol=1e-5)
    assert torch.allclose(att(sw)[:, 1], att(base)[:, 0], atol=1e-5)
    assert torch.allclose(att(sw)[:, Z:], att(base)[:, Z:], atol=1e-5)  # frontline attackers untouched
    # enemy backline 0 <-> 1: target columns 0/1 swap
    sw = net.policy_logits(_swap_slots(x, H + 2 * Z, H + 2 * Z + 1))
    assert torch.allclose(att(sw)[:, :, 0], att(base)[:, :, 1], atol=1e-5)
    assert torch.allclose(att(sw)[:, :, 1], att(base)[:, :, 0], atol=1e-5)
    assert torch.allclose(att(sw)[:, :, 2 * Z], att(base)[:, :, 2 * Z], atol=1e-5)  # base column


def test_frontline_slots_are_both_attackers_and_targets():
    """Frontline slot j is attacker row Z+j (when I hold it) and target column Z+j (when the enemy does)."""
    net = _perturbed_entity_net()
    present = ENC.split(X).mask
    ok = present[:, H + Z:H + Z + 2].all(1) & present[:, H] & present[:, H + 2 * Z]
    x = X[ok][:8]  # 2 frontline units; my and the enemy's backline slot 0 occupied (differ from slot 1)
    assert len(x) >= 2
    att = lambda logits: logits[:, SP.ATTACK0:].reshape(-1, 2 * Z, 2 * Z + 1)  # noqa: E731
    base = att(net.policy_logits(x))
    sw = att(net.policy_logits(_swap_slots(x, H + Z, H + Z + 1)))
    rows, cols = list(range(2 * Z)), list(range(2 * Z + 1))
    rows[Z], rows[Z + 1] = Z + 1, Z
    cols[Z], cols[Z + 1] = Z + 1, Z
    assert torch.allclose(sw[:, rows][:, :, cols], base, atol=1e-5)


def test_base_token_is_built_from_the_enemy_base_hp():
    net = NETS["entity"]()
    seen = []
    net.base_enc.register_forward_hook(lambda module, inp, out: seen.append(inp[0].detach().clone()))
    x = X[:16].clone()
    opp, mine = GLOBAL_FEATURES.index("opp_base_hp"), GLOBAL_FEATURES.index("my_base_hp")
    x[:, opp] = torch.linspace(0.05, 1.0, len(x))  # enemy and own base HP differ in every row
    x[:, mine] = torch.linspace(1.0, 0.05, len(x)) + 0.01
    net.policy_logits(x)
    assert torch.equal(seen[0][:, 0], x[:, opp]) and (seen[0][:, 1] == 1).all()


def test_checkpoint_loading_rules(tmp_path):
    net = NETS["entity"]()
    good = {"model": net.state_dict(), "net": net.spec(), "update": 1, "env_steps": 0, "args": {}}
    path = tmp_path / "good.pt"
    torch.save(good, path)
    agent = PPOAgent.from_checkpoint(str(path), CONFIG, deterministic=True)
    g = Game(CONFIG)
    g.reset(3)
    while not g.done:
        la = g.legal_actions()
        a = agent.act(g.observe(g.current_player()), la)
        assert a in la
        g.step(a)
    stage1 = {"model": {}, "net": {"obs_dim": 393, "n_actions": 71, "hidden": [256, 256]}}
    with pytest.raises(CheckpointError, match="Stage 1"):
        net_from_checkpoint(stage1, ENC)
    spec = net.spec()
    spec["layout"] = dict(spec["layout"], fingerprint="0" * 16)
    other = build_net(spec)
    with pytest.raises(CheckpointError, match="card pool"):
        net_from_checkpoint({"model": other.state_dict(), "net": spec}, ENC)
    net_from_checkpoint({"model": other.state_dict(), "net": spec}, ENC, allow_pool_mismatch=True)
