"""Stage 3 networks (SPEC §7): masking, shapes, gradients, token wiring of every action block
(permutation tests), previews, privileged critic, belief head, checkpoints.

Inputs are real v5 encodings: random games on the shipped content (mulligan on and off, fixed and
random decks), random games on a `build_ruleset` effect pool (CHOICE states with a pending token and
choose previews), and hand-built positions. The expected action <-> token wiring is written from the
SPEC §3/§6 numbers, not read from the network.
"""
from __future__ import annotations

import inspect
import json
import random

import numpy as np
import pytest
import torch

from cardgame.cards import build_ruleset, load_ruleset, sample_deal
from cardgame.engine import CHOICE, MAIN, MULLIGAN, Game
from cardgame.features import ENCODER_VERSION, GLOBAL_FEATURES, ObservationEncoder
from cardgame.rl.agent import CheckpointError, PPOAgent, net_from_checkpoint
from cardgame.rl.network import (MASK_VALUE, PolicyValueNet, PooledPolicyNet, TransformerPolicyNet,
                                 build_net)
from conftest import add_unit, blank_game, set_hand

CONFIG_M = load_ruleset()                 # shipped content, mulligan on (the training config)
CONFIG = load_ruleset(mulligan=False)     # same fingerprint and layout
ENC = ObservationEncoder(CONFIG_M)
LAY = ENC.layout()
assert ObservationEncoder(CONFIG).layout() == LAY

# SPEC §3 / §6 numbers (H = 10, Z = 5, R = 20)
H, Z = 10, 5
N = 154
PLAY0, MOVE0, ATTACK0, CHOOSE0, MULLIGAN0, CONFIRM = 1, 11, 16, 126, 143, 153
NT = 2 * Z + 1
G_TOK, HAND0, MYB0, FRONT0, OPPB0, MYBASE, OPPBASE, PEND = 0, 1, 11, 16, 21, 26, 27, 28
GF = {k: i for i, k in enumerate(GLOBAL_FEATURES)}


# ---------------------------------------------------------------- effect pool with chosen targets
def _unit(cid, atk, hp, cost=1, nature="troop", effects=()):
    return {"id": cid, "name": cid, "type": "unit", "nature": nature, "cost": cost, "attack": atk, "health": hp,
            "effects": list(effects)}


def _op(cid, effects, cost=1):
    return {"id": cid, "name": cid, "type": "operation", "cost": cost, "effects": list(effects)}


def _chosen(action, side, kind="unit", **params):
    e = {"trigger": "on_play", "action": action, "target": {"select": "chosen", "side": side, "kind": kind}}
    e.update(params)
    return e


def effect_rules(mulligan: bool):
    filler = [_unit(f"f{i:02d}", 1 + i % 3, 1 + i % 4, cost=1 + i % 6, nature=("troop", "fast", "ranged")[i % 3])
              for i in range(12)]
    ops = [_op("zap", [_chosen("damage", "any", "unit_or_base", amount=2)]),
           _op("mend", [_chosen("heal", "friendly", "unit_or_base", amount=2)]),
           _op("doom", [_chosen("destroy", "enemy")], cost=2),
           _op("rally", [_chosen("buff", "friendly", atk=1, hp=1)]),
           _op("snare", [_chosen("pin", "enemy")])]
    sniper = _unit("sniper", 2, 2, cost=2, effects=[
        {"trigger": "on_deploy", "action": "damage", "amount": 1,
         "target": {"select": "chosen", "side": "enemy", "kind": "unit_or_base"}}])
    ids = [c["id"] for c in filler]
    deck_a = {cid: 2 for cid in ids[:8]}
    deck_a.update({"zap": 3, "mend": 3, "doom": 3, "rally": 3, "sniper": 3})
    deck_a.update({ids[8]: 3, ids[9]: 2, ids[10]: 2, ids[11]: 2})
    deck_b = {cid: 2 for cid in ids}
    deck_b.update({"zap": 3, "snare": 3, "doom": 3, "rally": 3, "sniper": 3, "mend": 1})
    assert sum(deck_a.values()) == sum(deck_b.values()) == 40
    return build_ruleset(filler + ops + [sniper], [{"name": "a", "cards": deck_a}, {"name": "b", "cards": deck_b}],
                         mulligan=mulligan)


RULES_M, RULES = effect_rules(True), effect_rules(False)
ENC_R = ObservationEncoder(RULES_M)
LAY_R = ENC_R.layout()
assert LAY_R["n_cards"] != LAY["n_cards"]


# ---------------------------------------------------------------- data
def play_random(config, enc, n_games, seed, random_decks=False):
    """(X, M, OPP) of every decision of random games: float16-rounded encodings with the mover's mask,
    legal masks and the opponent's true hand counts. PLAY is preferred so that choices open."""
    g, rng = Game(config), random.Random(seed)
    X, M, OPP = [], [], []
    for s in range(n_games):
        decks = sample_deal(seed * 1000 + s, config, 0.7) if random_decks else None
        g.reset(seed * 1000 + s, decks)
        while not g.done:
            p = g.current_player()
            m = g.legal_mask()
            X.append(enc.encode(g.observe(p), m))
            M.append(m)
            OPP.append(np.bincount(g.hands[1 - p], minlength=enc.n_cards))
            la = g.legal_actions()
            plays = [a for a in la if PLAY0 <= a < MOVE0]
            g.step(plays[rng.randrange(len(plays))] if plays and rng.random() < 0.5 else la[rng.randrange(len(la))])
    X = np.stack(X).astype(np.float16).astype(np.float32)  # the rollout protocol
    return torch.from_numpy(X), torch.from_numpy(np.stack(M)), torch.from_numpy(np.stack(OPP).astype(np.float32))


def _cat(*parts):
    return tuple(torch.cat(t) for t in zip(*parts))


X, M, OPP = _cat(play_random(CONFIG_M, ENC, 6, 1), play_random(CONFIG, ENC, 4, 2),
                 play_random(CONFIG_M, ENC, 4, 3, random_decks=True))
XR, MR, OPPR = _cat(play_random(RULES_M, ENC_R, 6, 4), play_random(RULES, ENC_R, 6, 5))
PHASE = X[:, GF["phase_mulligan"]:GF["phase_choice"] + 1].argmax(1)
PHASE_R = XR[:, GF["phase_mulligan"]:GF["phase_choice"] + 1].argmax(1)


def test_data_covers_every_phase_and_action_block():
    assert (PHASE == MULLIGAN).sum() > 20 and (PHASE == MAIN).sum() > 200
    assert (PHASE_R == CHOICE).sum() > 20 and (PHASE_R == MULLIGAN).sum() > 20
    both = torch.cat([M, MR])
    for lo, hi in ((0, 1), (PLAY0, MOVE0), (MOVE0, ATTACK0), (ATTACK0, CHOOSE0), (CHOOSE0, MULLIGAN0),
                   (MULLIGAN0, CONFIRM), (CONFIRM, N)):
        assert both[:, lo:hi].any(), (lo, hi)
    assert ENC_R.split(XR).present[:, PEND].any() and ENC_R.split(XR).choose_preview.abs().sum() > 0


# ---------------------------------------------------------------- nets under test (small)
def make(kind, layout=LAY, **kw):
    if kind == "transformer":
        return TransformerPolicyNet(layout, d_model=32, layers=2, heads=4, ff=64, id_dim=8, pair_dim=16, **kw)
    if kind == "transformer_shared":
        return TransformerPolicyNet(layout, d_model=32, layers=1, heads=2, ff=64, id_dim=8, pair_dim=16,
                                    shared_trunk=True, **kw)
    if kind == "pooled":
        return PooledPolicyNet(layout, d_model=32, ctx_dim=64, id_dim=8, pair_dim=16, **kw)
    if kind == "mlp":
        return PolicyValueNet.from_layout(layout, hidden=(64, 64), **kw)
    raise KeyError(kind)


KINDS = ("transformer", "transformer_shared", "pooled", "mlp")
TOKEN_KINDS = ("transformer", "pooled")


def perturbed(kind, layout=LAY, seed=0, **kw):
    """A net with non-trivial heads (the output layers start near zero)."""
    torch.manual_seed(seed)
    net = make(kind, layout, **kw)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return net


def bce(logits, opp):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, (opp > 0).float())


# ================================================================ sampling, shapes, gradients
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("data", ["shipped", "effects"])
def test_masked_sampling_shapes_and_gradients(kind, data):
    x, m, opp, lay = (X, M, OPP, LAY) if data == "shipped" else (XR, MR, OPPR, LAY_R)
    torch.manual_seed(0)
    net = make(kind, lay)
    assert (net.kind, net.obs_dim, net.n_actions, net.n_cards) == \
        (kind.split("_")[0], lay["dim"], N, lay["n_cards"])
    gen = torch.Generator().manual_seed(1)
    for _ in range(3):
        a, logp = net.act(x, m, generator=gen)
        assert a.shape == logp.shape == (len(x),) and a.dtype == torch.int64
        assert m[torch.arange(len(a)), a].all() and torch.isfinite(logp).all() and (logp <= 0).all()
    lp, ent, v, bl = net.evaluate(x, m, a)
    assert torch.allclose(lp, logp, atol=1e-5)
    assert lp.shape == ent.shape == v.shape == (len(x),) and bl.shape == (len(x), lay["n_cards"])
    assert torch.isfinite(ent).all() and (ent >= -1e-6).all() and torch.isfinite(v).all()
    one_legal = m.sum(1) == 1
    assert one_legal.any() and (ent[one_legal].abs() < 1e-6).all() and (lp[one_legal].abs() < 1e-6).all()
    loss = -lp.mean() + v.pow(2).mean() - 0.01 * ent.mean() + bce(bl, opp)
    loss.backward()
    grads = {n: p.grad for n, p in net.named_parameters()}
    assert all(g is not None for g in grads.values()), [n for n, g in grads.items() if g is None]
    assert all(torch.isfinite(g).all() for g in grads.values())
    d, dlogp = net.act(x, m, deterministic=True)
    assert m[torch.arange(len(d)), d].all()
    assert torch.equal(d, net.policy_logits(x, m).argmax(1))


@pytest.mark.parametrize("kind", KINDS)
def test_logits_layout_and_masking(kind):
    net = make(kind)
    raw, masked = net.policy_logits(X[:64]), net.policy_logits(X[:64], M[:64])
    assert raw.shape == masked.shape == (64, N)
    assert (masked[~M[:64]] == MASK_VALUE).all() and torch.equal(masked[M[:64]], raw[M[:64]])
    assert torch.isfinite(raw).all()
    # near-uniform initial policy (small output layers)
    assert raw.abs().max() < 0.5
    lp = torch.log_softmax(masked, -1)
    n_legal = M[:64].sum(1, keepdim=True).float()
    assert torch.allclose(lp.exp()[M[:64]], (1 / n_legal).expand(-1, N)[M[:64]], atol=0.05)


@pytest.mark.parametrize("kind", KINDS)
def test_spec_round_trip(kind):
    net = perturbed(kind, LAY_R)
    spec = json.loads(json.dumps(net.spec()))  # JSON-safe
    assert spec == net.spec() and spec["kind"] == net.kind
    clone = build_net(spec)
    clone.load_state_dict(net.state_dict())
    assert clone.spec() == net.spec()
    x, m = XR[:40], MR[:40]
    assert torch.allclose(clone.policy_logits(x, m), net.policy_logits(x, m))
    assert torch.allclose(clone.value(x), net.value(x))
    assert torch.allclose(clone.belief_logits(x), net.belief_logits(x))


def test_build_net_rejects_old_and_unknown_specs():
    with pytest.raises(ValueError, match="Stage 1"):
        build_net({"obs_dim": 393, "n_actions": 71, "hidden": [256, 256]})
    with pytest.raises(ValueError, match="Stage 2"):
        build_net({"kind": "entity", "layout": {"version": 4}, "d_model": 128})
    with pytest.raises(ValueError, match="Stage 2"):
        build_net({"kind": "mlp", "obs_dim": 2000, "n_actions": 126, "hidden": [256, 256]})
    with pytest.raises(ValueError, match="unknown network kind"):
        build_net({"kind": "lstm"})
    stage2_layout = {"G": 30, "E": 20, "F": 40, "P": 4, "H": 10, "Z": 5, "n_cards": 25, "dim": 1500, "n_actions": 126,
                     "fingerprint": "0" * 16}
    with pytest.raises(ValueError, match="not a Stage 3"):
        build_net({"kind": "transformer", "layout": stage2_layout})
    with pytest.raises(ValueError, match="shared_trunk"):
        TransformerPolicyNet(LAY, shared_trunk=True, privileged_critic=True)
    with pytest.raises(ValueError, match="shared_trunk"):
        PooledPolicyNet(LAY, shared_trunk=True, privileged_critic=True)


def test_shared_trunk_shares_the_tower():
    shared = make("transformer_shared")
    assert shared.value_tower is shared.policy_tower
    sep = make("transformer")
    assert sep.value_tower is not sep.policy_tower
    assert not ({id(p) for p in sep.policy_tower.parameters()} & {id(p) for p in sep.value_tower.parameters()})
    # separate towers: the value loss does not reach the policy tower
    sep.value(X[:32]).pow(2).mean().backward()
    assert all(p.grad is None for p in sep.policy_tower.parameters())
    assert all(p.grad is None for p in sep.scorer.parameters())


@pytest.mark.parametrize("kind", KINDS)
def test_empty_boards_and_blank_inputs_give_finite_outputs(kind):
    net = make(kind)
    blank = torch.zeros(3, ENC.dim)  # nothing present at all: the always-present tokens are forced on
    mask = torch.zeros(3, N, dtype=torch.bool)
    mask[:, 0] = True
    rows = (PHASE == MULLIGAN) & ~ENC.split(X).present[:, MYB0:OPPB0 + Z].any(1)
    assert rows.sum() > 5
    for x, m in ((blank, mask), (X[rows], M[rows])):
        logits, value = net(x, m)
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()
        assert torch.isfinite(net.belief_logits(x)).all()
        a, logp = net.act(x, m)
        assert m[torch.arange(len(a)), a].all() and torch.isfinite(logp).all()


@pytest.mark.parametrize("kind", TOKEN_KINDS)
def test_absent_tokens_are_invisible(kind):
    """Key padding (Transformer) / masked pooling: whatever an absent token row holds never reaches a
    legal logit, the value or the belief."""
    net = perturbed(kind, LAY_R)
    x, m = XR[:150], MR[:150]
    p = ENC_R.split(x)
    off, T, F = LAY_R["offsets"], LAY_R["T"], LAY_R["F"]
    absent = ~p.present
    assert absent[:, PEND].any() and absent[:, HAND0 + H - 1].any() and absent[:, OPPB0 + Z - 1].any()
    y = x.clone()
    gen = torch.Generator().manual_seed(0)
    feats = y[:, off["features"]:off["ids"]].view(len(x), T, F)
    feats[absent] = torch.rand(int(absent.sum()), F, generator=gen)
    ids = y[:, off["ids"]:off["present"]]
    ids[absent] = torch.randint(1, LAY_R["n_cards"] + 1, (int(absent.sum()),), generator=gen).float()
    a, b = net.policy_logits(x, m), net.policy_logits(y, m)
    assert torch.allclose(a[m], b[m], atol=1e-5) and torch.allclose(net.value(x), net.value(y), atol=1e-5)
    assert torch.allclose(net.belief_logits(x), net.belief_logits(y), atol=1e-5)


def test_token_packing_matches_the_dense_encoder():
    """The Transformer tower packs each row's present tokens to the front and truncates the batch to its
    longest row; outputs equal the dense key-padded encoder over all T tokens."""
    net = perturbed("transformer", LAY_R)
    tower = net.policy_tower
    x = torch.cat([XR[:100], HB_XR])
    parts = net._parts(x)
    present = parts.present
    order = tower.pack_order(present)
    counts = present.sum(1)
    assert order.shape == (len(x), int(counts.max())) and order.shape[1] < LAY_R["T"]
    for r in range(len(x)):
        k = int(counts[r])
        assert order[r, :k].tolist() == present[r].nonzero().flatten().tolist()
        assert not present[r, order[r, k:]].any()
    for grad in (True, False):
        with torch.set_grad_enabled(grad):
            h, g = tower(parts)
            dense = tower.encoder(tower.embed(parts), src_key_padding_mask=~present) * present.unsqueeze(-1)
        assert torch.allclose(h, dense, atol=1e-5) and torch.equal(g, h[:, G_TOK])


@pytest.mark.parametrize("kind", TOKEN_KINDS)
def test_train_and_inference_modes_agree(kind):
    """Workers act under eval + inference_mode (Transformer fast path); the learner trains in train mode."""
    net = perturbed(kind, LAY_R)
    x, m = XR[:200], MR[:200]
    net.train()
    train = net.policy_logits(x, m)
    net.eval()
    with torch.inference_mode():
        infer = net.policy_logits(x, m)
        single = torch.cat([net.policy_logits(x[i:i + 1], m[i:i + 1]) for i in range(5)])
        v_infer = net.value(x)
    assert torch.allclose(train, infer, atol=1e-4)
    assert torch.allclose(single, infer[:5], atol=1e-4)
    net.train()
    assert torch.allclose(net.value(x), v_infer, atol=1e-4)


# ================================================================ token wiring (permutation tests)
def att_tok(a):
    return MYB0 + a if a < Z else FRONT0 + a - Z


def tgt_tok(t):
    return OPPB0 + t if t < Z else (FRONT0 + t - Z if t < 2 * Z else OPPBASE)


def choose_tok(t):
    return tgt_tok(t) if t <= 2 * Z else (MYB0 + t - 2 * Z - 1 if t <= 3 * Z else MYBASE)


def action_roles():
    """SPEC wiring: action index -> (kind, source token, target token or None)."""
    roles = {0: ("end", G_TOK, None), CONFIRM: ("confirm", G_TOK, None)}
    for i in range(H):
        roles[PLAY0 + i] = ("play", HAND0 + i, None)
        roles[MULLIGAN0 + i] = ("mulligan", HAND0 + i, None)
    for j in range(Z):
        roles[MOVE0 + j] = ("move", MYB0 + j, None)
    for a in range(2 * Z):
        for t in range(NT):
            roles[ATTACK0 + a * NT + t] = ("attack", att_tok(a), tgt_tok(t))
    for t in range(3 * Z + 2):
        roles[CHOOSE0 + t] = ("choose", PEND, choose_tok(t))
    assert sorted(roles) == list(range(N))
    return roles


ROLES = action_roles()
assert LAY["attacker_tokens"] == [att_tok(a) for a in range(2 * Z)]
assert LAY["attack_target_tokens"] == [tgt_tok(t) for t in range(NT)]
assert LAY["choose_tokens"] == [choose_tok(t) for t in range(3 * Z + 2)]


def action_perm(ti, tj):
    """perm with logits(swapped)[n] == logits(original)[perm[n]] when tokens ti and tj are swapped."""
    sig = lambda t: tj if t == ti else (ti if t == tj else t)  # noqa: E731
    index = {r: n for n, r in ROLES.items()}
    return torch.tensor([index[(k, sig(s), None if t is None else sig(t))] for k, s, t in
                         (ROLES[n] for n in range(N))])


def swap_tokens(x, ti, tj, lay):
    """Swap two tokens of the same group: feature rows, ids, presence, and their attack-preview
    rows/columns and choose-preview slots."""
    y = x.clone()
    off, T, F = lay["offsets"], lay["T"], lay["F"]
    B = len(x)
    feats = y[:, off["features"]:off["ids"]].view(B, T, F)
    for o in (off["ids"], off["present"]):
        y[:, [o + ti, o + tj]] = x[:, [o + tj, o + ti]]
    feats[:, [ti, tj]] = feats[:, [tj, ti]].clone()
    ap = y[:, off["attack_preview"]:off["choose_preview"]].view(B, 2 * Z, NT, 4)
    att, tgt, ch = lay["attacker_tokens"], lay["attack_target_tokens"], lay["choose_tokens"]
    if ti in att and tj in att:
        a, b = att.index(ti), att.index(tj)
        ap[:, [a, b]] = ap[:, [b, a]].clone()
    if ti in tgt and tj in tgt:
        a, b = tgt.index(ti), tgt.index(tj)
        ap[:, :, [a, b]] = ap[:, :, [b, a]].clone()
    cp = y[:, off["choose_preview"]:].view(B, 3 * Z + 2, 4)
    if ti in ch and tj in ch:
        a, b = ch.index(ti), ch.index(tj)
        cp[:, [a, b]] = cp[:, [b, a]].clone()
    return y


def swap_mask(m, perm):
    return m[:, perm]


def _choice_position(config, op_name, my_front=False):
    """A CHOICE state: `op_name` was played with two own backline units, two enemy backline units and
    two frontline units (mine if `my_front`, else the enemy's) on the board."""
    g = blank_game(current=0, first=0, round_=5, coins=10, config=config)
    set_hand(g, 0, [op_name, "f03", "f07"])
    add_unit(g, 0, "back", "f00", atk=2, hp=2, max_hp=5)
    add_unit(g, 0, "back", "f01", atk=1, hp=1)
    add_unit(g, 1, "back", "f02", atk=2, hp=3)
    add_unit(g, 1, "back", "f04", atk=5, hp=2, max_hp=4)
    owner = 0 if my_front else 1
    add_unit(g, owner, "front", "f05", atk=4, hp=4)
    add_unit(g, owner, "front", "f06", atk=1, hp=3, max_hp=6)
    g.base_hp = [12, 3]
    g.invalidate()
    g.step(PLAY0 + g.hands[0].index(config.cards.by_id(op_name).index))
    assert g.phase == CHOICE
    return g


def _main_position(my_front):
    """A MAIN state of the shipped pool with 3 hand cards, 2 units in every zone that matters."""
    units = sorted((c for c in CONFIG.cards.cards if c.is_unit and not c.token), key=lambda c: (c.cost, c.index))
    g = blank_game(current=0, first=0, round_=6, coins=6)
    set_hand(g, 0, [units[0].index, units[3].index, units[-1].index])
    add_unit(g, 0, "back", units[1].id)
    add_unit(g, 0, "back", units[4].id, hp=1)
    add_unit(g, 1, "back", units[2].id)
    add_unit(g, 1, "back", units[5].id)
    owner = 0 if my_front else 1
    add_unit(g, owner, "front", units[6].id)
    add_unit(g, owner, "front", units[7].id, atk=1)
    g.invalidate()
    return g


def _encode_states(games, enc):
    X_, M_ = [], []
    for g in games:
        m = g.legal_mask()
        X_.append(enc.encode(g.observe(g.current_player()), m))
        M_.append(m)
    return torch.from_numpy(np.stack(X_)), torch.from_numpy(np.stack(M_))


HB_X, HB_M = _encode_states([_main_position(True), _main_position(False)], ENC)
HB_XR, HB_MR = _encode_states([_choice_position(RULES, op, mf) for op in ("zap", "rally", "doom")
                               for mf in (True, False)], ENC_R)


def _rows(x, m, enc, *tokens):
    """(x, mask) rows where every given token is present, capped at 12."""
    keep = enc.split(x).present[:, list(tokens)].all(1)
    return x[keep][:12], m[keep][:12]


def _check_perm(net, x, m, lay, ti, tj, must_differ):
    base = net.policy_logits(x)
    perm = action_perm(ti, tj)
    sw = net.policy_logits(swap_tokens(x, ti, tj, lay))
    assert torch.allclose(sw, base[:, perm], atol=1e-5), (ti, tj, (sw - base[:, perm]).abs().max())
    # the masked logits permute the same way (the legal mask of the swapped state is the permuted mask)
    assert torch.allclose(net.policy_logits(swap_tokens(x, ti, tj, lay), swap_mask(m, perm)),
                          net.policy_logits(x, m)[:, perm], atol=1e-5)
    for n in must_differ:  # non-vacuous: the swapped actions really scored differently
        assert (base[:, n] - base[:, perm[n]]).abs().max() > 1e-4, n


@pytest.mark.parametrize("kind", TOKEN_KINDS)
def test_hand_tokens_wire_play_and_mulligan(kind):
    net = perturbed(kind)
    xr, mr = _rows(X, M, ENC, HAND0, HAND0 + 2)
    x, m = torch.cat([HB_X, xr]), torch.cat([HB_M, mr])
    _check_perm(net, x, m, LAY, HAND0, HAND0 + 2, [PLAY0, PLAY0 + 2, MULLIGAN0, MULLIGAN0 + 2])
    perm = action_perm(HAND0, HAND0 + 2)
    assert perm[PLAY0] == PLAY0 + 2 and perm[MULLIGAN0 + 2] == MULLIGAN0 and perm[PLAY0 + 1] == PLAY0 + 1
    assert torch.equal(perm[MOVE0:MULLIGAN0], torch.arange(MOVE0, MULLIGAN0))  # MOVE/ATTACK/CHOOSE untouched
    assert perm[0] == 0 and perm[CONFIRM] == CONFIRM


@pytest.mark.parametrize("kind", TOKEN_KINDS)
def test_own_backline_tokens_wire_move_attacker_rows_and_own_choose_slots(kind):
    net = perturbed(kind)
    i, j = MYB0, MYB0 + 1
    _check_perm(net, HB_X, HB_M, LAY, i, j, [MOVE0, ATTACK0 + 2 * Z, ATTACK0 + NT + 2 * Z])
    perm = action_perm(i, j)
    assert perm[MOVE0] == MOVE0 + 1 and perm[ATTACK0 + 3] == ATTACK0 + NT + 3
    assert perm[CHOOSE0 + 2 * Z + 1] == CHOOSE0 + 2 * Z + 2
    # in CHOICE states the own-backline CHOOSE slots permute (effect pool, pending token present)
    net_r = perturbed(kind, LAY_R)
    x = torch.cat([HB_XR, _rows(XR, MR, ENC_R, PEND, i, j)[0]])
    _check_perm(net_r, x, torch.ones(len(x), N, dtype=torch.bool), LAY_R, i, j,
                [CHOOSE0 + 2 * Z + 1, CHOOSE0 + 2 * Z + 2])


@pytest.mark.parametrize("kind", TOKEN_KINDS)
def test_enemy_backline_tokens_wire_attack_columns_and_choose_slots(kind):
    net = perturbed(kind)
    i, j = OPPB0, OPPB0 + 1
    _check_perm(net, HB_X, HB_M, LAY, i, j, [ATTACK0, ATTACK0 + 1])
    perm = action_perm(i, j)
    assert perm[ATTACK0 + 3 * NT] == ATTACK0 + 3 * NT + 1 and perm[ATTACK0 + 3 * NT + 2 * Z] == ATTACK0 + 3 * NT + 2 * Z
    net_r = perturbed(kind, LAY_R)
    x = torch.cat([HB_XR, _rows(XR, MR, ENC_R, PEND, i, j)[0]])
    _check_perm(net_r, x, torch.ones(len(x), N, dtype=torch.bool), LAY_R, i, j, [CHOOSE0, CHOOSE0 + 1])


@pytest.mark.parametrize("kind", TOKEN_KINDS)
def test_frontline_tokens_are_attackers_when_mine_and_targets_when_the_enemys(kind):
    net = perturbed(kind)
    i, j = FRONT0, FRONT0 + 1
    front_mine = HB_X[:, GF["front_mine"]] == 1
    assert front_mine.tolist() == [True, False]
    att = HB_M[:, ATTACK0:CHOOSE0].reshape(-1, 2 * Z, NT)
    assert att[0, Z:Z + 2].any() and not att[0, :, Z:2 * Z].any()   # mine: front units attack
    assert att[1, :, Z:Z + 2].any() and not att[1, Z:].any()        # enemy's: front units are targets
    _check_perm(net, HB_X, HB_M, LAY, i, j, [ATTACK0 + Z * NT, ATTACK0 + Z])
    perm = action_perm(i, j)
    rows = perm[ATTACK0:CHOOSE0].reshape(2 * Z, NT) - ATTACK0
    assert rows[Z, 0] == (Z + 1) * NT and rows[0, Z] == Z + 1 and rows[Z, Z] == (Z + 1) * NT + Z + 1
    masked = net.policy_logits(HB_X, HB_M)
    sw = net.policy_logits(swap_tokens(HB_X, i, j, LAY), swap_mask(HB_M, perm))
    legal = HB_M & (torch.arange(N) >= ATTACK0) & (torch.arange(N) < CHOOSE0)
    assert torch.allclose(sw[legal], masked[:, perm][legal], atol=1e-5)
    net_r = perturbed(kind, LAY_R)
    _check_perm(net_r, HB_XR, torch.ones(len(HB_XR), N, dtype=torch.bool), LAY_R, i, j, [CHOOSE0 + Z])


NULL_OF = {"end": "END_TURN", "play": "PLAY", "move": "MOVE", "mulligan": "MULLIGAN", "confirm": "CONFIRM"}


@pytest.mark.parametrize("kind", TOKEN_KINDS)
def test_scorer_reads_exactly_the_spec_tokens(kind):
    """Each logit depends on exactly its SPEC source and target tokens (plus g); null-target types each
    have their own learned null vector."""
    from cardgame.rl.network import NULL_TARGET_TYPES
    net = perturbed(kind, LAY_R)
    parts = net._parts(HB_XR[:1])
    torch.manual_seed(1)
    h = torch.randn(1, LAY_R["T"], net.d_model, requires_grad=True)
    g = torch.randn(1, net.scorer.p_g.in_features)
    logits = net.scorer(h, g, parts)
    assert logits.shape == (1, N)
    for n in range(N):
        gh, gn = torch.autograd.grad(logits[0, n], (h, net.scorer.null), retain_graph=True)
        kind_, s, t = ROLES[n]
        touched = set(gh[0].abs().sum(-1).nonzero().flatten().tolist())
        assert touched == {s} | ({t} if t is not None else set()), (n, kind_, touched)
        nulls = set(gn.abs().sum(-1).nonzero().flatten().tolist())
        assert nulls == ({NULL_TARGET_TYPES.index(NULL_OF[kind_])} if t is None else set()), (n, kind_, nulls)


@pytest.mark.parametrize("kind", TOKEN_KINDS)
def test_random_states_are_equivariant_for_every_group(kind):
    net = perturbed(kind, LAY_R, seed=3)
    for ti, tj in ((HAND0 + 1, HAND0 + 3), (MYB0, MYB0 + 2), (FRONT0, FRONT0 + 2), (OPPB0 + 1, OPPB0 + 2)):
        x, _ = _rows(XR, MR, ENC_R, ti, tj)
        if len(x):
            base, perm = net.policy_logits(x), action_perm(ti, tj)
            assert torch.allclose(net.policy_logits(swap_tokens(x, ti, tj, LAY_R)), base[:, perm], atol=1e-5)


# ================================================================ previews
@pytest.mark.parametrize("kind", TOKEN_KINDS)
def test_attack_previews_reach_their_attack_logit_only(kind):
    net = perturbed(kind)
    off = LAY["offsets"]["attack_preview"]
    base = net.policy_logits(HB_X)
    legal = HB_M[0, ATTACK0:CHOOSE0].nonzero().flatten()
    assert len(legal) >= 2
    for n in legal[:3].tolist():
        for k in range(4):
            x = HB_X.clone()
            x[0, off + n * 4 + k] += 1.0
            out = net.policy_logits(x)
            changed = (out - base).abs() > 1e-6
            assert changed[0, ATTACK0 + n] and changed.sum() == 1, (n, k)
            assert torch.equal(net.value(x), net.value(HB_X))  # previews feed the scorer only


@pytest.mark.parametrize("kind", TOKEN_KINDS)
def test_choose_previews_reach_their_choose_logit_only(kind):
    net = perturbed(kind, LAY_R)
    off = LAY_R["offsets"]["choose_preview"]
    x0, m0 = HB_XR[:1], HB_MR[:1]
    assert ENC_R.split(x0).choose_preview.abs().sum() > 0  # zap: damage previews
    base = net.policy_logits(x0)
    for t in m0[0, CHOOSE0:MULLIGAN0].nonzero().flatten()[:4].tolist():
        for k in range(4):
            x = x0.clone()
            x[0, off + t * 4 + k] += 1.0
            changed = (net.policy_logits(x) - base).abs() > 1e-6
            assert changed[0, CHOOSE0 + t] and changed.sum() == 1, (t, k)


# ================================================================ privileged critic and belief head
@pytest.mark.parametrize("kind", ("transformer", "pooled", "mlp"))
def test_privileged_critic_sees_the_opponent_hand_and_the_actor_never_does(kind):
    net = perturbed(kind, LAY_R, privileged_critic=True)
    x, m, opp = XR[:80], MR[:80], OPPR[:80]
    other = opp.roll(1, dims=0) + torch.eye(LAY_R["n_cards"])[torch.arange(80) % LAY_R["n_cards"]]
    with pytest.raises(ValueError, match="priv"):
        net.value(x)
    with pytest.raises(ValueError, match="priv"):
        net.evaluate(x, m, m.float().argmax(1))
    with pytest.raises(ValueError, match="shape"):
        net.value(x, opp[:, :3])
    v1, v2 = net.value(x, opp), net.value(x, other)
    assert torch.isfinite(v1).all() and (v1 - v2).abs().max() > 1e-4
    for fn in (net.policy_logits, net.act, net.belief_logits):
        assert "priv" not in inspect.signature(fn).parameters
    a, _ = net.act(x, m)
    e1, e2 = net.evaluate(x, m, a, priv=opp), net.evaluate(x, m, a, priv=other)
    assert torch.equal(e1[0], e2[0]) and torch.equal(e1[1], e2[1]) and torch.equal(e1[3], e2[3])
    assert torch.allclose(e1[2], v1) and not torch.allclose(e1[2], e2[2])
    if kind != "mlp":
        assert net.policy_tower.embed.priv_in is None and net.value_tower.embed.priv_in is not None
    # the gradient of the value loss never reaches what the actor uses
    net.zero_grad()
    net.value(x, opp).pow(2).mean().backward()
    actor = (list(net.policy_tower.parameters()) + list(net.scorer.parameters())) if kind != "mlp" \
        else list(net.policy_body.parameters()) + list(net.policy_out.parameters())
    assert all(p.grad is None or not p.grad.any() for p in actor)
    # a net without a privileged critic ignores priv
    plain = perturbed(kind, LAY_R)
    assert torch.equal(plain.value(x, opp), plain.value(x))


@pytest.mark.parametrize("kind", ("transformer", "pooled", "mlp"))
def test_belief_head(kind):
    net = perturbed(kind, LAY_R)
    b = net.belief_logits(XR[:50])
    assert b.shape == (50, LAY_R["n_cards"]) and torch.isfinite(b).all()
    a, _ = net.act(XR[:50], MR[:50])
    assert torch.allclose(net.evaluate(XR[:50], MR[:50], a)[3], b)
    bce(b, OPPR[:50]).backward()
    head = net.belief_head.weight.grad
    assert head is not None and head.abs().sum() > 0
    off = make(kind, LAY_R, belief=False)
    assert off.belief_logits(XR[:5]) is None and off.evaluate(XR[:5], MR[:5], a[:5])[3] is None
    assert not any("belief" in k for k in off.state_dict())
    assert off.spec()["belief"] is False and build_net(off.spec()).belief_logits(XR[:2]) is None
    # belief=False removes exactly the head
    n_on = sum(p.numel() for p in make(kind, LAY_R).parameters())
    n_off = sum(p.numel() for p in off.parameters())
    assert n_on - n_off == net.belief_head.weight.numel() + net.belief_head.bias.numel()


def test_card_vectors_follow_the_card_table():
    net = make("transformer")
    emb = net.policy_tower.embed
    cv = emb.cardvecs()
    assert cv.shape == (LAY["n_cards"] + 1, 32) and not cv[0].any()
    assert torch.equal(emb.card_table[1:], torch.tensor(LAY["card_table"]))
    # the deck token reads the remaining deck through the card vectors
    x = X[:8].clone()
    off = LAY["offsets"]["deck_counts"]
    before = net.value(x)
    x[:, off] += 2.0
    assert (net.value(x) - before).abs().max() > 0


# ================================================================ checkpoints and PPOAgent
def _ckpt(net):
    return {"model": net.state_dict(), "net": net.spec(), "update": 1, "env_steps": 0, "args": {}}


def _play_full_game(agent, config, seed):
    g = Game(config)
    g.reset(seed)
    phases = set()
    while not g.done:
        la = g.legal_actions()
        phases.add(g.phase)
        a = agent.act(g.observe(g.current_player()), la)
        assert a in la
        g.step(a)
    return phases


def test_checkpoint_plays_full_games_through_ppo_agent(tmp_path):
    for kind, config in (("transformer", CONFIG_M), ("pooled", CONFIG_M), ("mlp", CONFIG)):
        torch.manual_seed(0)
        net = make(kind, privileged_critic=kind == "transformer")  # the agent never needs priv
        path = tmp_path / f"{kind}.pt"
        torch.save(_ckpt(net), path)
        for deterministic in (True, False):
            agent = PPOAgent.from_checkpoint(str(path), config, deterministic=deterministic, seed=1)
            assert not agent.net.training
            phases = _play_full_game(agent, config, seed=3)
            assert MAIN in phases and (MULLIGAN in phases) == config.mulligan
    # effect pool: CHOICE states through the agent
    net = make("transformer", LAY_R)
    path = tmp_path / "rules.pt"
    torch.save(_ckpt(net), path)
    agent = PPOAgent.from_checkpoint(str(path), RULES_M, seed=2)
    seen = set()
    for seed in range(4):
        seen |= _play_full_game(agent, RULES_M, seed)
    assert {MULLIGAN, MAIN, CHOICE} <= seen


def test_ppo_agent_matches_the_rollout_protocol():
    """PPOAgent's probabilities are the network's softmax over the legal actions of the float16-rounded
    encoding made with the observer's mask."""
    torch.manual_seed(0)
    net = perturbed("transformer", LAY_R)
    agent = PPOAgent(net, RULES_M, seed=0)
    g = _choice_position(RULES_M, "zap")
    obs, mask = g.observe(0), g.legal_mask()
    la = g.legal_actions()
    x = torch.from_numpy(ENC_R.encode(obs, mask).astype(np.float16).astype(np.float32))[None]
    with torch.inference_mode():
        want = torch.softmax(net.policy_logits(x, torch.from_numpy(mask)[None])[0][la].double(), 0).numpy()
    assert np.allclose(agent.action_probs(obs, la), want, atol=1e-6)
    det = PPOAgent(net, RULES_M, deterministic=True)
    assert det.act(obs, la) == la[int(np.argmax(want))]


def test_checkpoint_loading_rules():
    net = make("transformer")
    good = _ckpt(net)
    loaded = net_from_checkpoint(good, ENC)
    assert not loaded.training and loaded.spec() == net.spec()
    # Stage 1: no network kind
    stage1 = {"model": {}, "net": {"obs_dim": 393, "n_actions": 71, "hidden": [256, 256]}}
    with pytest.raises(CheckpointError, match="Stage 1"):
        net_from_checkpoint(stage1, ENC)
    # Stage 2: the entity network, a Stage 2 MLP (no encoder version), an encoder-v4 layout
    stage2 = {"model": {}, "net": {"kind": "entity", "layout": {"version": 4, "E": 20}, "d_model": 128}}
    with pytest.raises(CheckpointError, match="Stage 2"):
        net_from_checkpoint(stage2, ENC)
    stage2_mlp = {"model": {}, "net": {"kind": "mlp", "obs_dim": 1500, "n_actions": 126, "hidden": [256, 256]}}
    with pytest.raises(CheckpointError, match="Stage 2"):
        net_from_checkpoint(stage2_mlp, ENC)
    v4 = dict(good, net=dict(net.spec(), layout=dict(LAY, version=ENCODER_VERSION - 1)))
    with pytest.raises(CheckpointError, match="encoder version"):
        net_from_checkpoint(v4, ENC)
    with pytest.raises(CheckpointError, match="not a PPO checkpoint"):
        net_from_checkpoint({"net": net.spec()}, ENC)
    # another card pool / decks (fingerprint), with and without allow_pool_mismatch
    spec = net.spec()
    spec["layout"] = dict(spec["layout"], fingerprint="0" * 16)
    other = {"model": build_net(spec).state_dict(), "net": spec}
    with pytest.raises(CheckpointError, match="card pool"):
        net_from_checkpoint(other, ENC)
    rebased = net_from_checkpoint(other, ENC, allow_pool_mismatch=True)
    assert rebased.layout["fingerprint"] == ENC.fingerprint
    mlp = make("mlp")
    mspec = dict(mlp.spec(), fingerprint="0" * 16)
    with pytest.raises(CheckpointError, match="card pool"):
        net_from_checkpoint({"model": mlp.state_dict(), "net": mspec}, ENC)
    assert net_from_checkpoint({"model": mlp.state_dict(), "net": mspec}, ENC, allow_pool_mismatch=True).fingerprint \
        == ENC.fingerprint
    # a different input size (other pool size) cannot load, even with allow_pool_mismatch
    rules_ckpt = _ckpt(make("transformer", LAY_R))
    for allow in (False, True):
        with pytest.raises(CheckpointError, match="obs_dim"):
            net_from_checkpoint(rules_ckpt, ENC, allow_pool_mismatch=allow)
    # weights that do not fit the spec
    wrong = dict(good, model=make("pooled").state_dict())
    with pytest.raises(CheckpointError, match="do not fit"):
        net_from_checkpoint(wrong, ENC)
    # PPOAgent refuses a net built for another encoding
    with pytest.raises(CheckpointError, match="obs_dim"):
        PPOAgent(make("transformer", LAY_R), CONFIG)
