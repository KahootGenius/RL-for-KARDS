"""PPO learner + rollout workers (Stage 3, SPEC §8): deals, opponents, belief labels, privileged critic,
micro-batched learning, quick eval vs lookahead, best.pt, checkpoints, failures, Ctrl-C, the CLI and the
batched inference server (§8.1).

Everything runs with tiny configs (in-process workers unless the test is about processes, one
Transformer layer, d_model 32, small batches) on the CPU.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import random
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as tF

from cardgame.agents.lookahead_agent import LookaheadAgent
from cardgame.cards import deck_rng, generate_deck, load_ruleset, sample_deal
from cardgame.engine import DRAW, MAIN, Game
from cardgame.features import GLOBAL_FEATURES, HAND_SCALE
from cardgame.rl import ppo
from cardgame.rl.agent import CheckpointError
from cardgame.rl.ppo import (BEST_MIN_CELL, QUICK_EVAL_SEED_BASE, TRAIN_SEED_BASE, TRAIN_SEEDS_PER_RUN, PPOConfig,
                             PPOTrainer, best_score, resolve_inference_server, select_device)
from cardgame.rl.rollout import (OPPONENT_KINDS, POLICY_KEY, TIME_KEYS, InferenceServer, RolloutWorker,
                                 RolloutWorkerError, SparseRows, WorkerInit, WorkerPool, _Env, _Opponent)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_ruleset()
ARCHS = ("transformer", "pooled", "mlp")
KILLED = -15 if sys.platform == "win32" else -9  # Process.kill() exit code: Windows multiprocessing reports -SIGTERM


def tiny(tmp_path, **kw) -> PPOConfig:
    base = dict(run_dir=str(tmp_path / "run"), workers=0, envs_per_worker=8, batch_steps=256, minibatch=128,
                micro_batch=64, epochs=2, d_model=32, layers=1, heads=2, ff=64, id_dim=4, ctx_dim=32, pair_dim=8,
                hidden=(32,), snapshot_every=1, max_snapshots=2, eval_every=0, device="cpu", torch_threads=1,
                tensorboard=False, select_games=0)
    base.update(kw)
    return PPOConfig(**base)


def worker_init(tr: PPOTrainer, worker_id: int = 0, n_workers: int = 1, **kw) -> WorkerInit:
    cfg = tr.cfg
    init = WorkerInit(worker_id=worker_id, n_workers=n_workers, seed_base=tr.seed_base,
                      seed_limit=tr.seed_block + TRAIN_SEEDS_PER_RUN, config=CONFIG, net_spec=tr.net.spec(),
                      envs=cfg.envs_per_worker, self_play_prob=cfg.self_play_prob, opp_weights=dict(cfg.opp_weights),
                      snapshots_per_worker=cfg.snapshots_per_worker, random_deck_frac=cfg.random_deck_frac)
    return dataclasses.replace(init, **kw)


def _unpack(tr, t) -> np.ndarray:
    return np.unpackbits(t["mask"], axis=1, count=tr.n_actions).astype(bool)


# ---------------------------------------------------------------- collection and learning, every arch
@pytest.mark.parametrize("arch", ARCHS)
def test_collect_and_learn_in_process(tmp_path, arch):
    tr = PPOTrainer(tiny(tmp_path, arch=arch))
    assert tr.n_actions == tr.encoder.n_actions == 154
    trajs, stats = tr.collect()
    n = sum(len(t["act"]) for t in trajs)
    assert n >= tr.cfg.batch_steps and stats["transitions"] == n
    for t in trajs:
        T = len(t["act"])
        mask = _unpack(tr, t)
        assert t["mask"].shape == (T, (tr.n_actions + 7) // 8)
        assert t["obs"].dtype == np.float16 and t["obs"].shape == (T, tr.encoder.dim)
        assert t["opp_hand"].dtype == np.uint8 and t["opp_hand"].shape == (T, tr.n_cards)
        assert mask[np.arange(T), t["act"]].all(), "recorded action was not legal"
        assert (mask.sum(1) > 1).all(), "forced decisions are not recorded"
        assert (t["obs"][:, 0] == 1).all(), "every recorded state is the learner's own turn"
        assert t["reward"] in (-1.0, 0.0, 1.0)
    # the worker time breakdown covers the whole collect
    times = stats["time"]
    assert set(times) == set(TIME_KEYS) and all(v >= 0 for v in times.values())
    assert sum(times.values()) == pytest.approx(stats["worker_seconds"], rel=0.05, abs=0.02)
    assert times["infer"] > 0 and times["encode"] > 0 and times["engine"] > 0
    out = tr.learn(trajs)
    assert all(np.isfinite(v) for v in out.values())
    assert {"pg_loss", "v_loss", "entropy", "kl", "belief_loss", "belief_rprec", "belief_acc"} <= set(out)
    for k, v in tr.net.state_dict().items():  # the CPU actor is refreshed after every update
        assert torch.equal(v.cpu(), tr.actor.state_dict()[k])


def test_training_row_logs_the_worker_time_breakdown(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, total_updates=1))
    tr.train(log=lambda *_: None)
    row = json.loads((tr.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    for key in TIME_KEYS:
        assert f"wt_{key}_s" in row and f"wt_{key}_frac" in row
    assert sum(row[f"wt_{k}_frac"] for k in TIME_KEYS) == pytest.approx(1.0, abs=1e-3)
    assert row["infer_batch_mean"] >= 1 and "belief_loss" in row and "belief_rprec" in row


def test_opp_hand_labels_are_the_engine_hands(tmp_path, monkeypatch):
    """Every recorded opp_hand is the opponent's true hand at that decision (counted independently here)."""
    tr = PPOTrainer(tiny(tmp_path, self_play_prob=0.5))
    worker = RolloutWorker(worker_init(tr))
    expected = {}
    orig = RolloutWorker._act

    def spy(self, net, idxs, actions, record):
        if record:
            for i in idxs:
                g = self.envs[i].game
                c = Counter(g.hands[1 - g.current])
                expected.setdefault((g.seed, g.current), []).append([c[k] for k in range(tr.n_cards)])
        return orig(self, net, idxs, actions, record)

    monkeypatch.setattr(RolloutWorker, "_act", spy)
    out = worker.collect(300)
    assert out["trajs"]
    gi = GLOBAL_FEATURES.index("opp_hand_size")
    nonempty = 0
    for t in out["trajs"]:
        assert t["opp_hand"].tolist() == expected[(t["seed"], t["seat"])]
        sizes = np.rint(t["obs"][:, gi].astype(np.float32) * HAND_SCALE)  # the public hand size, from the encoding
        assert np.array_equal(sizes, t["opp_hand"].sum(1))
        nonempty += int((t["opp_hand"].sum(1) > 0).sum())
    assert nonempty > 0


def test_deal_mix_follows_random_deck_frac(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    for frac in (0.0, 0.7, 1.0):
        worker = RolloutWorker(worker_init(tr, random_deck_frac=frac))
        env = worker.envs[0]
        random_seats = [0, 0]
        n = 1500 if frac == 0.7 else 100
        for _ in range(n):
            worker._reset_env(env)
            g = env.game
            want = sample_deal(g.seed, CONFIG, frac)
            for seat in (0, 1):
                if isinstance(want[seat], int):
                    assert g.deck_ids[seat] == want[seat]
                else:
                    assert g.deck_ids[seat] == -1 and g.decklists[seat] == tuple(sorted(want[seat]))
                random_seats[seat] += g.deck_ids[seat] == -1
        for seat in (0, 1):
            assert abs(random_seats[seat] / n - frac) < 0.04, (frac, random_seats)


@pytest.mark.parametrize("kind", OPPONENT_KINDS)
def test_every_opponent_kind_plays(tmp_path, monkeypatch, kind):
    tr = PPOTrainer(tiny(tmp_path))
    worker = RolloutWorker(worker_init(tr, self_play_prob=0.0, opp_weights={kind: 1.0}))
    calls = []
    if kind == "snapshot":
        worker.update_pool(add={"s": (tr.net.spec(), tr._weights())}, names=["s"])
        snap = worker.snapshots["s"]
        policy_logits = snap.policy_logits
        snap.policy_logits = lambda *a, **kw: (calls.append(len(a[0])), policy_logits(*a, **kw))[1]
    elif kind == "lookahead":
        orig = LookaheadAgent.act_game

        def act_game(self, game, player):
            assert player == game.current_player()
            calls.append(player)
            return orig(self, game, player)

        monkeypatch.setattr(LookaheadAgent, "act_game", act_game)
    else:
        agent = worker.scripted[kind].agent
        orig = agent.act
        agent.act = lambda obs, legal: (calls.append(len(legal)), orig(obs, legal))[1]
    out = worker.collect(200)
    assert calls, f"the {kind} opponent never chose an action"
    assert out["results"] and {r[0] for r in out["results"]} == {kind}
    if kind != "snapshot":
        assert out["time"]["scripted"] > 0
    # results carry the deck ids; generated decks are -1 and the learner keeps fixed lookahead pairs apart
    for _, my_deck, opp_deck, _ in out["results"]:
        assert -1 <= my_deck < CONFIG.n_decks and -1 <= opp_deck < CONFIG.n_decks
    for row in out["results"]:
        tr._record_result(*row)
    if kind == "lookahead":
        assert all(i >= 0 and j >= 0 for i, j in tr.lookahead_pairs)
        n_fixed = sum(1 for r in out["results"] if r[1] >= 0 and r[2] >= 0)
        assert sum(len(v) for v in tr.lookahead_pairs.values()) == n_fixed
        n_rand = sum(1 for r in out["results"] if r[1] < 0 and r[2] < 0)
        assert len(tr.results["lookahead_randdeck"]) == n_rand


def test_opponent_mix_follows_the_config(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    worker = RolloutWorker(worker_init(tr))
    worker.pool_names = worker.active = ["snap_a"]
    kinds = Counter(worker._sample_opponent()[0].kind for _ in range(20000))
    assert abs(kinds["self"] / 20000 - 0.5) < 0.02
    rest = 20000 - kinds["self"]
    for kind, w in (("lookahead", 0.15), ("random", 0.05), ("snapshot", 0.30)):
        assert abs(kinds[kind] / rest - w / 0.5) < 0.03, kinds
    assert kinds["greedy"] == 0  # an allowed kind with weight 0 by default
    worker.active = []
    assert "snapshot" not in Counter(worker._sample_opponent()[0].kind for _ in range(2000))


def test_scripted_opponents_have_their_own_rng_streams(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    worker = RolloutWorker(worker_init(tr))
    mine = [worker.rng.random() for _ in range(4)]
    for kind in ("random", "lookahead"):
        rng = RolloutWorker(worker_init(tr)).scripted[kind].agent.rng
        assert [rng.random() for _ in range(4)] != mine
    other = RolloutWorker(worker_init(tr, worker_id=1, n_workers=2)).scripted["lookahead"].agent.rng
    assert other.random() != RolloutWorker(worker_init(tr)).scripted["lookahead"].agent.rng.random()


def test_every_policy_only_observes_its_own_seat(tmp_path, monkeypatch):
    tr = PPOTrainer(tiny(tmp_path, opp_weights={"lookahead": 0.2, "random": 0.1, "greedy": 0.1, "snapshot": 0.3}))
    tr._add_snapshot(tr.run_dir / "x.pt")
    calls = []
    orig = Game.observe

    def spy(self, player):
        calls.append((player, self.current))
        return orig(self, player)

    monkeypatch.setattr(Game, "observe", spy)
    tr.collect()
    tr.quick_eval(2)
    assert calls and all(p == cur for p, cur in calls)


# ---------------------------------------------------------------- learner math
def test_gae_uses_recomputed_values(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, gamma=0.9, gae_lambda=0.8))
    vals = torch.tensor([0.1, -0.2, 0.3])
    tr.net.value = lambda obs, priv=None: vals[:len(obs)]  # the learner's value net, not the worker's
    traj = {"obs": np.zeros((3, tr.encoder.dim), np.float16),
            "mask": np.packbits(np.ones((3, tr.n_actions), bool), axis=1),
            "act": np.zeros(3, np.int16), "logp": np.zeros(3, np.float32), "version": np.zeros(3, np.int32),
            "opp_hand": np.zeros((3, tr.n_cards), np.uint8), "reward": 1.0}
    b = tr._batch([traj])
    g, lam, v = 0.9, 0.8, vals.numpy()
    d2 = 1.0 - v[2]
    d1 = g * v[2] - v[1]
    d0 = g * v[1] - v[0]
    a2 = d2
    a1 = d1 + g * lam * a2
    a0 = d0 + g * lam * a1
    assert np.allclose(b["adv"].numpy(), [a0, a1, a2], atol=1e-6)
    assert np.allclose(b["ret"].numpy(), np.array([a0, a1, a2]) + v, atol=1e-6)
    assert b["mask"].shape == (3, tr.n_actions) and b["mask"].all()


def test_values_are_recomputed_in_row_order(tmp_path):
    """Values are computed on rows sorted by token count, in chunks, and must land back in row order."""
    tr = PPOTrainer(tiny(tmp_path, micro_batch=8))  # value chunks of 32 rows
    trajs, _ = tr.collect()
    b = tr._batch(trajs)
    with torch.no_grad():
        direct = tr.net.value(b["obs"].float()).numpy()
    assert np.allclose(b["val"].numpy(), direct, atol=1e-5)
    assert len(set(b["n_present"].tolist())) > 1


def _bandit(tr, reward_a=1.0, reward_b=-1.0, n=64, logp_shift=(0.0, 0.0)):
    """n one-step trajectories taking legal action A (reward_a) and n taking B (reward_b) in one position.

    Behaviour log-probs are the current policy's, shifted by `logp_shift` (A, B) to set the PPO ratio."""
    g, rng = Game(CONFIG), random.Random(1)
    g.reset(5)
    while len(g.legal_actions()) < 3 or g.phase != MAIN:
        la = g.legal_actions()
        g.step(la[rng.randrange(len(la))])
    m = g.legal_mask()
    x = tr.encoder.encode(g.observe(g.current_player()), m).astype(np.float16)
    a_act, b_act = (int(i) for i in np.flatnonzero(m)[1:3])
    xt, mt = torch.from_numpy(x.astype(np.float32))[None], torch.from_numpy(m)[None]
    with torch.no_grad():
        lp = torch.log_softmax(tr.net.policy_logits(xt, mt), -1)[0]
    opp = np.bincount(g.hands[1 - g.current], minlength=tr.n_cards).astype(np.uint8)
    trajs = []
    for act, r, shift in ((a_act, reward_a, logp_shift[0]), (b_act, reward_b, logp_shift[1])):
        trajs += [{"obs": x[None], "mask": np.packbits(m[None], axis=1), "act": np.array([act], np.int16),
                   "logp": np.array([float(lp[act]) + shift], np.float32), "version": np.zeros(1, np.int32),
                   "opp_hand": opp[None], "reward": r} for _ in range(n)]
    return trajs, xt, mt, a_act, b_act


def _zero_values(*trainers):
    for tr in trainers:  # value 0 everywhere: advantage = reward, and the value loss has no gradient
        tr.net.value = lambda obs, priv=None: torch.zeros(len(obs))


def test_update_favours_the_positive_advantage_action(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, epochs=4, minibatch=32, target_kl=0.0))
    trajs, xt, mt, a, b = _bandit(tr)
    with torch.no_grad():
        before = tr.net.policy_logits(xt, mt)[0]
    tr.learn(trajs)
    with torch.no_grad():
        after = tr.net.policy_logits(xt, mt)[0]
    assert after[a] - after[b] > before[a] - before[b] + 1e-3


def test_update_is_invariant_to_a_reward_shift(tmp_path):
    cfg = tiny(tmp_path, vf_coef=0.0, minibatch=1024, target_kl=0.0)
    tr1, tr2 = PPOTrainer(cfg), PPOTrainer(cfg)
    _zero_values(tr1, tr2)
    t1, *_ = _bandit(tr1, 1.0, -1.0)
    t2, *_ = _bandit(tr2, 6.0, 4.0)  # same rewards + 5: identical normalised advantages
    for tr, trajs in ((tr1, t1), (tr2, t2)):
        torch.manual_seed(1)
        tr.learn(trajs)
    for (k, v1), v2 in zip(tr1.net.state_dict().items(), tr2.net.state_dict().values()):
        assert torch.allclose(v1, v2, atol=1e-6), k


def test_clipped_ratios_give_no_policy_gradient(tmp_path):
    # ratio e (A > 0) and 1/e (A < 0) lie beyond the clip range on the side PPO clips: no update at all
    tr = PPOTrainer(tiny(tmp_path, vf_coef=0.0, ent_coef=0.0, belief=False, epochs=1, minibatch=1024,
                         target_kl=0.0))
    _zero_values(tr)
    trajs, *_ = _bandit(tr, 1.0, -1.0, logp_shift=(-1.0, 1.0))
    before = {k: v.clone() for k, v in tr.net.state_dict().items()}
    tr.learn(trajs)
    for k, v in tr.net.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_entropy_bonus_raises_the_entropy(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, vf_coef=0.0, ent_coef=0.5, belief_coef=0.0, epochs=4, minibatch=32,
                         target_kl=0.0))
    _zero_values(tr)
    with torch.no_grad():
        for p in tr.net.parameters():  # start from a clearly non-uniform policy
            p.add_(torch.randn_like(p) * 0.3)
    trajs, xt, mt, *_ = _bandit(tr, 0.0, 0.0)  # zero advantages: only the entropy term acts

    def entropy():
        with torch.no_grad():
            lp = torch.log_softmax(tr.net.policy_logits(xt, mt), -1)
            return float(-(lp.exp() * lp).sum())

    e0 = entropy()
    tr.learn(trajs)
    assert entropy() > e0 + 0.05


def _batch_bce(tr, b) -> float:
    with torch.no_grad():
        logits = tr.net.belief_logits(b["obs"].float())
        return float(tF.binary_cross_entropy_with_logits(logits, (b["opp_hand"] > 0).float()))


def test_belief_loss_is_finite_and_decreases_on_a_fixed_batch(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, belief_coef=1.0, lr=3e-3, epochs=4, minibatch=128, target_kl=0.0))
    trajs, _ = tr.collect()
    b = tr._batch(trajs)
    first = tr.learn(trajs)
    assert np.isfinite(first["belief_loss"]) and 0 < first["belief_loss"] < 2
    assert 0 <= first["belief_rprec"] <= 1 and 0 <= first["belief_acc"] <= 1
    start = _batch_bce(tr, b)
    for _ in range(4):
        last = tr.learn(trajs)
    assert last["belief_loss"] < first["belief_loss"]
    assert _batch_bce(tr, b) < 0.8 * start


def test_belief_off_has_no_belief_loss(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, belief=False))
    assert tr.net.belief_head is None
    out = tr.learn(tr.collect()[0])
    assert not any(k.startswith("belief") for k in out)


@pytest.mark.parametrize("arch, priv", [("transformer", False), ("pooled", False), ("mlp", False),
                                        ("transformer", True)])
def test_micro_batch_accumulation_equals_the_full_batch_gradient(tmp_path, arch, priv):
    cfg = tiny(tmp_path, arch=arch, privileged_critic=priv, micro_batch=4096)
    full, micro = PPOTrainer(cfg), PPOTrainer(dataclasses.replace(cfg, micro_batch=24))
    micro.net.load_state_dict(full.net.state_dict())
    trajs, _ = full.collect()
    b = full._batch(trajs)
    data = full._device_data(b)
    idx = torch.randperm(len(b["act"]), generator=torch.Generator().manual_seed(0))[:200]
    s_full = full._minibatch_backward(data, idx, belief_stats=True)
    s_micro = micro._minibatch_backward(data, idx, belief_stats=True)
    assert s_full["finite"] and s_micro["finite"]
    for k in s_full:
        if k != "finite":  # rank/threshold statistics may flip on a float near-tie: one flip is ~1e-4
            tol = 2e-3 if k in ("belief_acc", "belief_rprec") else 1e-6
            assert s_micro[k] == pytest.approx(s_full[k], rel=1e-4, abs=tol), k
    n_checked = 0
    for (name, p1), p2 in zip(full.net.named_parameters(), micro.net.parameters()):
        if p1.grad is None:
            assert p2.grad is None or not p2.grad.any(), name
            continue
        assert torch.allclose(p1.grad, p2.grad, rtol=1e-4, atol=1e-6), name
        n_checked += 1
    assert n_checked > 5


def test_minibatches_are_sorted_by_present_tokens(tmp_path, monkeypatch):
    tr = PPOTrainer(tiny(tmp_path, micro_batch=16))
    trajs, _ = tr.collect()
    seen = []
    evaluate = tr.net.evaluate

    def spy(x, mask, actions, priv=None):
        seen.append(tr.present_counts(x))
        return evaluate(x, mask, actions, priv=priv)

    tr.net.evaluate = spy
    b = tr._batch(trajs)
    data = tr._device_data(b)
    tr._minibatch_backward(data, torch.arange(len(b["act"]))[:128])
    counts = torch.cat(seen)
    assert len(seen) == 8 and torch.equal(counts, counts.sort().values)
    # read from the encoder layout's token-present block
    lay = tr.encoder.layout()
    a = lay["offsets"]["present"]
    assert torch.equal(b["n_present"], (b["obs"][:, a:a + lay["T"]] > 0.5).sum(1))


def test_privileged_critic_values_depend_on_priv_and_the_actor_does_not(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, privileged_critic=True))
    trajs, _ = tr.collect()
    b1 = tr._batch(trajs)
    rng = np.random.default_rng(0)
    other = [dict(t, opp_hand=rng.integers(0, 3, size=t["opp_hand"].shape).astype(np.uint8)) for t in trajs]
    b2 = tr._batch(other)
    assert not torch.allclose(b1["val"], b2["val"]), "values ignore the opponent's hand"
    x, m, a = b1["obs"].float(), b1["mask"], b1["act"]
    e1 = tr.net.evaluate(x, m, a, priv=b1["opp_hand"].float())
    e2 = tr.net.evaluate(x, m, a, priv=b2["opp_hand"].float())
    assert torch.equal(e1[0], e2[0]) and torch.equal(e1[1], e2[1]) and torch.equal(e1[3], e2[3])
    assert not torch.allclose(e1[2], e2[2])
    out = tr.learn(trajs)
    assert all(np.isfinite(v) for v in out.values())
    with pytest.raises(ValueError):
        PPOTrainer(tiny(tmp_path, privileged_critic=True, shared_trunk=True))


def test_transitions_carry_the_policy_version(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    worker = RolloutWorker(worker_init(tr))
    worker.set_weights(7, None)
    out = worker.collect(64)
    assert out["trajs"] and all((t["version"] == 7).all() for t in out["trajs"])
    worker.set_weights(8, None)
    later = worker.collect(64)["trajs"]
    assert any(set(t["version"]) == {7, 8} for t in later), "games running across updates keep both versions"
    tr.update = 9
    stats = tr.learn(out["trajs"])
    assert stats["lag_mean"] == 2 and stats["lag_max"] == 2 and stats["stale_frac"] == 1.0


def test_select_device(monkeypatch):
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in ("cpu", "cuda", "mps")
    fallback = "mps" if torch.backends.mps.is_available() else "cpu"
    # a GPU this torch build has no kernels for (e.g. an RTX 50-series card, sm_120, under torch < 2.7)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA GeForce RTX 5080")
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_90"])
    ones = torch.ones

    def no_kernel(*args, device=None, **kw):
        if str(device).startswith("cuda"):
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return ones(*args, device=device, **kw)

    monkeypatch.setattr(torch, "ones", no_kernel)
    with pytest.warns(UserWarning, match="sm_120.*Falling back"):
        assert select_device("auto").type == fallback
    assert select_device("cuda").type == "cuda"  # an explicit device is never second-guessed
    monkeypatch.setattr(ppo, "_cuda_error", lambda: None)
    assert select_device("auto").type == "cuda"


@pytest.mark.parametrize("cuda_build, reason", [(None, "CPU-only build"), ("12.8", "sees none.*nvidia-smi")])
def test_select_device_explains_a_cpu_fallback(monkeypatch, cuda_build, reason):
    """No usable CUDA and no MPS: auto still picks the CPU but says why and how to fix it."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", cuda_build)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    with pytest.warns(UserWarning, match=f"training on the CPU because .*{reason}.*pip uninstall -y torch.*cu128"):
        assert select_device("auto").type == "cpu"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert select_device("cpu").type == "cpu"  # an explicit --device cpu is silent


def test_amp_is_cuda_only(tmp_path):
    assert PPOTrainer(tiny(tmp_path, amp=True)).amp is False  # bf16 autocast never runs on CPU/MPS


# ---------------------------------------------------------------- workers
def test_workers_deal_disjoint_seeds(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    seeds = set()
    for w in range(3):
        worker = RolloutWorker(worker_init(tr, w, 3))
        mine = [worker._next_seed() for _ in range(50)]
        assert mine == [tr.seed_base + w + 3 * n for n in range(50)]
        assert not seeds & set(mine)
        seeds |= set(mine)
    assert all(TRAIN_SEED_BASE <= s < QUICK_EVAL_SEED_BASE for s in seeds)


def test_in_process_worker_has_no_parent_to_watch(tmp_path):
    # the parent-alive check runs every 4096 env steps; in-process it must not fire
    tr = PPOTrainer(tiny(tmp_path, envs_per_worker=32, batch_steps=2000, self_play_prob=1.0))
    _, stats = tr.collect()
    assert stats["env_steps"] > 4096 and tr.pool.local.init.parent_pid == 0


def test_finished_games_reward_the_winner(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    worker = RolloutWorker(worker_init(tr))
    step = (np.zeros(tr.encoder.dim, np.float16), np.ones(tr.n_actions, bool), 0, 0.0, 0,
            np.zeros(tr.n_cards, np.uint8))
    for seats, opp in (((0,), worker.scripted["lookahead"]), ((1,), worker.scripted["random"]),
                       ((0, 1), _Opponent("self", "self"))):
        for winner in (0, 1, DRAW):
            g = Game(CONFIG)
            g.reset(0)
            g.done, g._winner = True, winner
            env = _Env(g, opp=opp, learner_seats=seats, traj={s: [step] for s in seats})
            out, results = [], []
            assert worker._finish(env, out, results) == len(seats)
            expected = [0.0 if winner == DRAW else 1.0 if winner == s else -1.0 for s in seats]
            assert [t["reward"] for t in out] == expected
            assert [t["seat"] for t in out] == list(seats) and all(t["seed"] == 0 for t in out)
            if opp.kind == "self":  # result logged from the first player's side
                assert results == [("self_first", g.deck_ids[g.first_player], g.deck_ids[1 - g.first_player],
                                    0.0 if winner == DRAW else 1.0 if winner == g.first_player else -1.0)]
            else:
                assert [r[3] for r in results] == expected and results[0][0] == opp.kind


def test_snapshot_pool_reaches_workers(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    tr.collect()
    tr._add_snapshot(tr.run_dir / "s.pt")
    tr.collect()
    worker = tr.pool.local
    assert worker.pool_names == ["snap_00000"] and "snap_00000" in worker.snapshots


def test_snapshots_leave_workers_once_dropped_from_the_pool(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    worker = RolloutWorker(worker_init(tr))
    worker.update_pool(add={"a": (tr.net.spec(), tr._weights())}, names=["a"])
    assert "a" in worker.snapshots
    worker.envs[0].opp = _Opponent("snapshot", "a")  # a running game still plays against it: kept
    worker.update_pool(remove=["a"], names=[])
    assert "a" in worker.snapshots
    worker.envs[0].opp = None
    worker.update_pool(names=[])
    assert "a" not in worker.snapshots


def test_spawned_workers_collect_and_report_errors(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, workers=2))
    try:
        trajs, stats = tr.collect()
        assert stats["transitions"] >= tr.cfg.batch_steps and trajs
        assert tr.worker_games[0] > 0 and tr.worker_games[1] > 0
        assert stats["n_workers"] == 2 and stats["time"]["infer"] > 0
    finally:
        tr.close()
    bad = worker_init(tr)
    bad.net_spec = {"kind": "nonsense"}
    pool = WorkerPool([bad], in_process=False, timeout=60)
    with pytest.raises(RolloutWorkerError, match="nonsense"):
        pool.collect(0, {}, [], 10)


def _start_collecting(pool: WorkerPool, weights: dict, quotas) -> None:
    for conn, quota in zip(pool.conns, quotas):
        conn.send(("collect", 0, weights, {"add": {}, "remove": [], "names": []}, quota))


def test_close_does_not_wait_for_busy_or_blocked_workers(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, envs_per_worker=64))
    pool = WorkerPool([worker_init(tr, w, 2) for w in range(2)], in_process=False, timeout=60)
    procs = list(pool.procs)
    try:
        # worker 0 collects forever; worker 1 finishes and blocks sending a result far larger than the pipe buffer
        _start_collecting(pool, tr._weights(), (10 ** 9, 2000))
        assert pool.conns[1].poll(120)
        t0 = time.time()
        pool.close()
        assert time.time() - t0 < 4.0  # one shared 2 s grace period, then terminate
        assert not any(p.is_alive() for p in procs)
    finally:
        pool.close(force=True)


def test_dead_workers_are_reported_with_their_exit_code(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, workers=2))
    try:
        tr.collect()
        procs = list(tr.pool.procs)
        procs[1].kill()  # e.g. the OOM killer between two updates: found when sending the next request
        procs[1].join(10)
        with pytest.raises(RolloutWorkerError, match=rf"rollout worker 1 .*exit code {KILLED}\b"):
            tr.collect()
        assert not any(p.is_alive() for p in procs)
    finally:
        tr.close(force=True)
    pool = WorkerPool([worker_init(tr, w, 2) for w in range(2)], in_process=False, timeout=60)
    procs = list(pool.procs)
    try:
        _start_collecting(pool, tr._weights(), (10 ** 9, 10 ** 9))
        procs[0].kill()  # dies mid-collection: found while waiting for the replies
        with pytest.raises(RolloutWorkerError, match=rf"rollout worker 0 .*exit code {KILLED}\b"):
            pool._gather()
        assert not any(p.is_alive() for p in procs)
    finally:
        pool.close(force=True)


@pytest.mark.filterwarnings("ignore:the training deal seeds")  # batch_steps 1e8: the projection warns
def test_interrupted_training_stops_busy_workers_at_once(tmp_path, monkeypatch):
    tr = PPOTrainer(tiny(tmp_path, workers=1, batch_steps=10 ** 8))  # workers would collect for hours
    seen = {}

    def interrupted(pool):
        seen.update(t=time.time(), procs=list(pool.procs))
        raise KeyboardInterrupt  # Ctrl-C while the learner waits for the workers

    monkeypatch.setattr(WorkerPool, "_gather", interrupted)
    with pytest.raises(KeyboardInterrupt):
        tr.train(log=lambda *_: None)
    assert time.time() - seen["t"] < 3.0
    assert tr.pool is None and not any(p.is_alive() for p in seen["procs"])


# ---------------------------------------------------------------- quick eval and best.pt
def test_quick_eval_vs_lookahead(tmp_path, monkeypatch):
    tr = PPOTrainer(tiny(tmp_path))
    resets = []
    orig = Game.reset

    def spy(self, seed, decks=None):
        resets.append((seed, decks))
        return orig(self, seed, decks)

    monkeypatch.setattr(Game, "reset", spy)
    n = CONFIG.n_decks
    n_deals = 2 * n * n  # half random deals, half covering every ordered fixed pair once
    res = tr.quick_eval(n_deals)
    assert res["games"] == 2 * n_deals and res["random_games"] == res["fixed_games"] == n_deals
    assert len(res["cells"]) == n * (n + 1) // 2 and res["min_cell"] == min(res["cells"].values())
    assert res["win_rate"] == pytest.approx((res["random_win_rate"] + res["fixed_win_rate"]) / 2)
    for key in ("win_rate", "draw_rate", "random_win_rate", "random_draw_rate", "fixed_win_rate", "min_cell"):
        assert 0 <= res[key] <= 1
    # deal d: seed QUICK_EVAL_SEED_BASE + d, played twice; random decks from deck_rng, then every fixed pair
    assert [s for s, _ in resets] == [QUICK_EVAL_SEED_BASE + d for d in range(n_deals) for _ in (0, 1)]
    half = n_deals // 2
    for d in (0, half - 1):
        seed = QUICK_EVAL_SEED_BASE + d
        assert resets[2 * d][1] == (generate_deck(deck_rng(seed, 0), CONFIG), generate_deck(deck_rng(seed, 1), CONFIG))
    assert [resets[2 * d][1] for d in range(half, n_deals)] == [divmod(k, n) for k in range(n * n)]
    # reproducible: per-game lookahead seeds and a fixed sampling seed
    assert tr.quick_eval(4) == tr.quick_eval(4)


def test_best_score_rule():
    hi = BEST_MIN_CELL
    eligible = {"win_rate": 0.5, "random_win_rate": 0.55, "min_cell": hi}
    strong_but_lopsided = {"win_rate": 0.95, "random_win_rate": 0.99, "min_cell": hi - 0.01}
    assert best_score(eligible) == (1, 0.55) and best_score(strong_but_lopsided) == (0, 0.95)
    assert best_score(eligible) > best_score(strong_but_lopsided)
    assert best_score(dict(eligible, random_win_rate=0.6)) > best_score(eligible)
    assert best_score({"win_rate": 0.7, "random_win_rate": None, "min_cell": 0.8}) == (1, 0.7)


def test_best_pt_follows_the_selection_rule(tmp_path, monkeypatch):
    evals = iter([  # (overall, random-deck, min cell) -> best.pt written?
        (0.50, 0.45, 0.40),  # (0, 0.50): first eval, written
        (0.60, 0.70, 0.50),  # (0, 0.60): better overall, written
        (0.55, 0.50, 0.65),  # (1, 0.50): first eligible eval beats any ineligible one, written
        (0.90, 0.95, 0.50),  # (0, 0.90): ineligible, kept out
        (0.50, 0.60, 0.70),  # (1, 0.60): higher random-deck rate among eligible, written
        (0.80, 0.55, 0.90),  # (1, 0.55): eligible but lower random-deck rate, kept out
    ])

    def fake_eval(self, n_deals, deterministic=False):
        w, r, c = next(evals)
        return {"win_rate": w, "draw_rate": 0.0, "games": 4, "random_win_rate": r, "random_draw_rate": 0.0,
                "random_games": 2, "fixed_win_rate": w, "fixed_games": 2, "min_cell": c, "cells": {"0-1": c}}

    monkeypatch.setattr(PPOTrainer, "quick_eval", fake_eval)
    tr = PPOTrainer(tiny(tmp_path, total_updates=6, eval_every=1, batch_steps=64, minibatch=64))
    tr.train(log=lambda *_: None)
    rows = [json.loads(line) for line in (tr.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r.get("best_update") for r in rows] == [1, 2, 3, None, 5, None]
    assert torch.load(tr.run_dir / "best.pt", weights_only=False)["update"] == 5
    assert tuple(torch.load(tr.run_dir / "latest.pt", weights_only=False)["best_eval"]) == (1, 0.6)
    assert rows[0]["eval_lookahead_random_win_rate"] == 0.45 and rows[0]["eval_lookahead_min_cell"] == 0.4


# ---------------------------------------------------------------- checkpoints, config, CLI
def test_checkpoint_resume(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, total_updates=3, batch_steps=128))
    tr.train(log=lambda *_: None)
    state = torch.load(tr.run_dir / "latest.pt", weights_only=False)
    assert state["update"] == 3 and [n for n, _ in state["snapshots"]] == ["snap_00002", "snap_00003"]
    assert state["net"]["layout"] == tr.encoder.layout() and state["encoder_version"] == 5
    tr2 = PPOTrainer(tiny(tmp_path, total_updates=5, batch_steps=128))
    tr2.resume(str(tr.run_dir / "latest.pt"))
    assert tr2.update == 3 and [n for n, _, _ in tr2.snapshots] == ["snap_00002", "snap_00003"]
    assert tr2.seed_base == state["seed_base"] + state["n_workers"] * (max(state["worker_games"]) + 1)
    assert tr2.lr_at(3) == pytest.approx(state["lr"])
    for k, v in tr.net.state_dict().items():
        assert torch.equal(v, tr2.net.state_dict()[k])
    with pytest.raises(ValueError, match="architecture"):
        PPOTrainer(tiny(tmp_path, d_model=16)).resume(str(tr.run_dir / "latest.pt"))
    with pytest.raises(ValueError, match="architecture"):
        PPOTrainer(tiny(tmp_path, arch="pooled")).resume(str(tr.run_dir / "latest.pt"))


def _stage2_checkpoints(tmp_path) -> list:
    """Checkpoints as Stage 2's train.py wrote them: the entity network and an MLP without encoder version."""
    args = dataclasses.asdict(PPOConfig())
    args.update(arch="entity", attention_layers=0, pair_mlp_dim=32, opp_weights={"greedy": 0.15, "random": 0.05,
                                                                                 "snapshot": 0.3})
    out = []
    for name, net in (("entity", {"kind": "entity", "layout": {"version": 4, "E": 20}, "d_model": 128}),
                      ("mlp", {"kind": "mlp", "obs_dim": 1500, "n_actions": 126, "hidden": [256, 256]})):
        path = tmp_path / f"stage2_{name}.pt"
        torch.save({"model": {}, "net": net, "update": 120, "args": dict(args), "optimizer": {}, "lr": 1e-4}, path)
        out.append(path)
    return out


def test_resume_refuses_stage2_checkpoints(tmp_path):
    train = importlib.import_module("train")
    for path in _stage2_checkpoints(tmp_path):
        with pytest.raises(CheckpointError, match="Stage 2.*cannot be resumed"):
            PPOTrainer(tiny(tmp_path)).resume(str(path))
        with pytest.raises(CheckpointError, match="Stage 2"):
            train.make_config(train.build_parser().parse_args(["--resume", str(path)]))
    # another card pool / deck list cannot be resumed either
    tr = PPOTrainer(tiny(tmp_path, total_updates=1, batch_steps=64, minibatch=64))
    tr.train(log=lambda *_: None)
    state = torch.load(tr.run_dir / "latest.pt", weights_only=False)
    state["net"]["layout"]["fingerprint"] = "0" * 16
    torch.save(state, tmp_path / "other_pool.pt")
    with pytest.raises(CheckpointError, match="card pool, deck list or encoder layout"):
        PPOTrainer(tiny(tmp_path)).resume(str(tmp_path / "other_pool.pt"))


def test_cli_refuses_a_stage2_checkpoint_clearly(tmp_path):
    path = _stage2_checkpoints(tmp_path)[0]
    env = dict(os.environ, PYTHONWARNINGS="ignore", PYTHONIOENCODING="cp1252")
    proc = subprocess.run([sys.executable, "train.py", "--resume", str(path), "--device", "cpu"], cwd=ROOT,
                          capture_output=True, text=True, env=env, timeout=300)
    assert proc.returncode == 2
    assert "Stage 2 checkpoint" in proc.stderr and "start a new run" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("bad", [{"seed": 100}, {"snapshot_every": 0}, {"envs_per_worker": 0}, {"minibatch": 0},
                                 {"micro_batch": 0}, {"eval_every": 1, "eval_deals": 0}, {"arch": "rnn"},
                                 {"arch": "entity"}, {"workers": -2}, {"self_play_prob": 1.5},
                                 {"random_deck_frac": 1.5}, {"random_deck_frac": -0.1}, {"belief_coef": -1.0},
                                 {"shared_trunk": True, "privileged_critic": True},
                                 {"shared_trunk": True, "arch": "mlp"}, {"heads": 3},
                                 {"opp_weights": {"lookahead": 0.2, "lookahaed": 0.1}},
                                 {"opp_weights": {"lookahead": 0.2, "random": -0.1}},
                                 {"opp_weights": {"lookahead": float("nan")}},
                                 {"opp_weights": {"lookahead": 0.0, "random": 0.0, "greedy": 0.0, "snapshot": 0.5}},
                                 {"inference_server": "yes"}, {"inference_server": True}])
def test_config_validation(tmp_path, bad):
    with pytest.raises(ValueError):
        PPOTrainer(tiny(tmp_path, **bad))


def test_default_config_matches_spec():
    cfg = PPOConfig()
    assert (cfg.arch, cfg.d_model, cfg.layers, cfg.heads, cfg.ff, cfg.id_dim, cfg.ctx_dim) == \
        ("transformer", 128, 3, 4, 256, 16, 256)
    assert cfg.belief and cfg.belief_coef == 0.25 and not cfg.privileged_critic and not cfg.shared_trunk
    assert cfg.random_deck_frac == 0.7 and cfg.self_play_prob == 0.5
    assert cfg.opp_weights == {"lookahead": 0.15, "random": 0.05, "snapshot": 0.30}
    assert cfg.micro_batch == 1024 and cfg.minibatch == 8192 and cfg.amp and cfg.eval_deals == 256
    assert cfg.inference_server == "auto" and cfg.serve_snapshots is False
    assert cfg.select_top == 3 and cfg.select_games == 2000


def test_cli_flags_and_merging(tmp_path):
    PPOTrainer(tiny(tmp_path, self_play_prob=1.0, opp_weights={"snapshot": 1.0}))  # pool never used: fine
    train = importlib.import_module("train")
    parser = train.build_parser()
    cfg = train.make_config(parser.parse_args(["--opp-weights", "lookahead=0.4,snapshot=0"]))
    assert cfg.opp_weights == {"lookahead": 0.4, "random": 0.05, "snapshot": 0.0}
    assert train.make_config(parser.parse_args([])) == PPOConfig()
    with pytest.raises(SystemExit):
        parser.parse_args(["--opp-weights", "lookahead"])
    cfg = train.make_config(parser.parse_args(["--no-belief", "--no-amp", "--privileged-critic", "--arch", "pooled",
                                               "--hidden", "64", "64", "--random-deck-frac", "0.5"]))
    assert (cfg.belief, cfg.amp, cfg.privileged_critic, cfg.arch, cfg.hidden, cfg.random_deck_frac) == \
        (False, False, True, "pooled", (64, 64), 0.5)
    tr = PPOTrainer(dataclasses.replace(tiny(tmp_path), belief=cfg.belief))
    assert tr.net.belief_head is None and tr.net.spec()["belief"] is False
    for mode in ("on", "off", "auto"):
        assert train.make_config(parser.parse_args(["--inference-server", mode])).inference_server == mode
    assert train.make_config(parser.parse_args([])).serve_snapshots is False
    assert train.make_config(parser.parse_args(["--serve-snapshots"])).serve_snapshots is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--inference-server", "gpu"])


def test_cli_help_lists_the_stage3_flags():
    env = dict(os.environ, PYTHONWARNINGS="ignore", PYTHONIOENCODING="cp1252")
    proc = subprocess.run([sys.executable, "train.py", "--help"], cwd=ROOT, capture_output=True, text=True, env=env,
                          timeout=300, check=True)
    for flag in ("--arch", "--layers", "--heads", "--ff", "--id-dim", "--ctx-dim", "--hidden", "--no-belief",
                 "--belief-coef", "--privileged-critic", "--shared-trunk", "--random-deck-frac", "--micro-batch",
                 "--no-amp", "--no-tensorboard", "--opp-weights", "--eval-deals", "--resume", "--updates",
                 "--inference-server", "--serve-snapshots"):
        assert flag in proc.stdout, flag
    assert "--belief " not in proc.stdout and "--amp " not in proc.stdout  # on by default: only --no-<name>


def test_train_cli_one_update_then_resume(tmp_path):
    pytest.importorskip("tensorboard")
    run = tmp_path / "cli"
    common = [sys.executable, "train.py", "--run-dir", str(run), "--workers", "0", "--envs-per-worker", "8",
              "--batch-steps", "128", "--minibatch", "64", "--micro-batch", "32", "--d-model", "32", "--layers", "1",
              "--heads", "2", "--ff", "64", "--id-dim", "4", "--pair-dim", "8", "--eval-every", "1",
              "--eval-deals", "2", "--device", "cpu", "--torch-threads", "1", "--select-games", "2"]
    env = dict(os.environ, PYTHONWARNINGS="ignore", PYTHONIOENCODING="cp1252")
    first = subprocess.run(common + ["--updates", "1", "--ent-coef", "0.02", "--opp-weights", "lookahead=0.3",
                                     "--no-belief"], cwd=ROOT, capture_output=True, text=True, env=env, timeout=600)
    assert first.returncode == 0, first.stderr
    assert "device=cpu" in first.stdout and "arch=transformer" in first.stdout
    assert "inference_server=off" in first.stdout  # auto on the CPU (and always with --workers 0)
    for name in ("metrics.jsonl", "latest.pt", "config.json", "best.pt", "selection.json", "cand_00001.pt"):
        assert (run / name).is_file(), name
    assert "selection: best.pt = cand_00001.pt" in first.stdout  # the final selection pass (SPEC 8)
    assert any((run / "tb").iterdir()), "TensorBoard event file written"
    info = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert info["belief"] is False and info["encoder_version"] == 5 and info["n_actions"] == 154
    resumed = subprocess.run([sys.executable, "train.py", "--resume", str(run / "latest.pt"), "--updates", "2",
                              "--opp-weights", "random=0.1", "--eval-every", "0"], cwd=ROOT, capture_output=True,
                             text=True, env=env, timeout=600)
    assert resumed.returncode == 0, resumed.stderr
    assert "resumed from" in resumed.stdout
    rows = [json.loads(line) for line in open(run / "metrics.jsonl", encoding="utf-8")]
    assert [r["update"] for r in rows] == [1, 2]
    assert "eval_lookahead_random_win_rate" in rows[0] and "wt_infer_frac" in rows[1]
    args = torch.load(run / "latest.pt", weights_only=False)["args"]
    assert args["ent_coef"] == 0.02 and args["batch_steps"] == 128 and args["total_updates"] == 2
    assert args["belief"] is False and args["eval_every"] == 0
    assert args["opp_weights"] == {"lookahead": 0.3, "random": 0.1, "snapshot": 0.3}  # merged over the saved run's


# ---------------------------------------------------------------- batched inference server (SPEC 8.1)
def test_inference_server_resolution(tmp_path):
    assert resolve_inference_server("auto", "cuda", 4) is True
    for dev in ("cpu", "mps"):
        assert resolve_inference_server("auto", dev, 4) is False  # auto = on iff the learner device is CUDA
        assert resolve_inference_server("on", dev, 4) is True
    assert resolve_inference_server("off", "cuda", 4) is False
    assert resolve_inference_server("on", "cuda", 0) is False  # in-process collection keeps local nets
    with pytest.raises(ValueError):
        resolve_inference_server("yes", "cpu", 4)
    assert PPOTrainer(tiny(tmp_path, workers=2)).inference_server is False  # auto on the CPU
    assert PPOTrainer(tiny(tmp_path, workers=2, inference_server="on")).inference_server is True
    tr = PPOTrainer(tiny(tmp_path, workers=0, inference_server="on"))  # ignored in-process, no error
    assert tr.inference_server is False
    trajs, stats = tr.collect()
    assert trajs and tr.pool.server is None and "server" not in stats and tr.pool.local.net is not None


class _ServerPipe:
    """A server-mode worker's end of the pipe, answered at once by an InferenceServer (in-process test)."""

    def __init__(self, server):
        self.server, self.reply, self.keys = server, None, []

    def send(self, msg):
        kind, key, obs, bits = msg  # obs travels sparse (SPEC 8.1)
        assert kind == "infer" and isinstance(obs, SparseRows) and obs.values.dtype == np.float16
        assert bits.dtype == np.uint8 and bits.shape == (obs.shape[0], (self.server.n_actions + 7) // 8)
        self.keys.append(key)
        self.reply = ("logits", self.server.serve([(key, obs, bits)])[0])

    def recv(self):
        reply, self.reply = self.reply, None
        return reply


def _same_trajectories(a: list, b: list, logp_atol: float = 0.0) -> None:
    assert len(a) == len(b) and a
    for x, y in zip(a, b):
        assert (x["seed"], x["seat"], x["reward"]) == (y["seed"], y["seat"], y["reward"])
        for key in ("obs", "mask", "act", "version", "opp_hand"):
            assert np.array_equal(x[key], y[key]), key
        if logp_atol:
            np.testing.assert_allclose(x["logp"], y["logp"], rtol=0, atol=logp_atol)
        else:
            assert np.array_equal(x["logp"], y["logp"])


@pytest.mark.parametrize("serve_snapshots", [True, False])
def test_server_mode_worker_holds_no_nets_and_plays_like_a_local_one(tmp_path, serve_snapshots):
    """Same seeds and the same batches: a server-mode worker records bit-identical trajectories. With
    serve_snapshots=False it runs its snapshots locally and only asks for the policy."""
    tr = PPOTrainer(tiny(tmp_path, self_play_prob=0.3, opp_weights={"lookahead": 0.05, "random": 0.05,
                                                                    "snapshot": 0.6}))
    spec = tr.net.spec()
    snap = PPOTrainer(tiny(tmp_path, seed=1))._weights()  # a snapshot that differs from the policy
    local = RolloutWorker(worker_init(tr))
    local.set_weights(0, tr._weights())
    server = InferenceServer(tr.net, "cpu", tr.n_actions, serve_snapshots=serve_snapshots)
    pipe = _ServerPipe(server)
    remote = RolloutWorker(worker_init(tr, inference_server=True, serve_snapshots=serve_snapshots), conn=pipe)
    assert remote.net is None
    for w in (local, remote):
        w.update_pool(add={"snap_a": (spec, snap)}, names=["snap_a"])
    assert ("snap_a" in remote.snapshots) is not serve_snapshots and "snap_a" in local.snapshots
    assert RolloutWorker(worker_init(tr, inference_server=True)).net is not None  # no pipe: local nets
    server.begin([("snap_a", spec, snap)])
    assert ("snap_a" in server.weights) is serve_snapshots
    a, b = local.collect(300), remote.collect(300)
    assert set(pipe.keys) == ({POLICY_KEY, "snap_a"} if serve_snapshots else {POLICY_KEY})
    _same_trajectories(a["trajs"], b["trajs"])
    assert a["results"] == b["results"] and "snapshot" in {r[0] for r in b["results"]}
    assert a["snapshots_in_use"] == b["snapshots_in_use"]
    assert b["infer_calls"] >= len(pipe.keys) and b["time"]["infer"] > 0


def test_inference_server_batches_by_network_and_keeps_snapshots_in_use(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    other = PPOTrainer(tiny(tmp_path, seed=1))
    spec = tr.net.spec()
    trajs = sorted(tr.collect()[0], key=lambda t: -len(t["act"]))
    server = InferenceServer(tr.net, "cpu", tr.n_actions, serve_snapshots=True)
    tr.net.train()
    server.begin([("snap_a", spec, other._weights())])
    assert not tr.net.training  # the policy serves in eval mode
    reqs = [(key, t["obs"][:k], t["mask"][:k]) for key, t, k in
            zip((POLICY_KEY, "snap_a", POLICY_KEY, "snap_a", POLICY_KEY), trajs, (3, 1, 5, 2, 4))]
    out = server.serve(reqs)
    for (key, obs, bits), logits in zip(reqs, out):
        net = tr.net if key == POLICY_KEY else other.net
        mask = np.unpackbits(bits, axis=1, count=tr.n_actions).astype(bool)
        with torch.no_grad():
            want = net.eval().policy_logits(torch.from_numpy(obs.astype(np.float32)), torch.from_numpy(mask))
        assert logits.dtype == np.float32 and logits.shape == (len(obs), tr.n_actions)
        np.testing.assert_allclose(logits, want.numpy(), rtol=0, atol=1e-5)
    st = server.stats
    assert (st["forwards"], st["requests"], st["rows"]) == (2, 5, 15) and st["forward_s"] > 0
    assert dict(server.rows_by_key) == {POLICY_KEY: 12, "snap_a": 3}
    # snapshots: kept while in the pool or played by a running game, then forgotten
    server.begin([("snap_b", spec, tr._weights())])  # snap_a left the trainer's pool
    server.retain({"snap_b", "snap_a"})              # ... but a running game still plays it
    assert "snap_a" in server.nets and server.serve([reqs[1]])[0].shape == (1, tr.n_actions)
    server.retain({"snap_b"})
    assert "snap_a" not in server.weights and "snap_a" not in server.nets
    with pytest.raises(RolloutWorkerError, match="unknown network 'snap_a'"):
        server.serve([reqs[1]])


def test_inference_server_matches_local_nets_with_spawned_workers(tmp_path, monkeypatch):
    """Two spawned workers in server mode play the same games as with local CPU nets (same seeds -> same
    observations and actions; log-probs up to float rounding of batches mixing both workers' rows),
    snapshot games included, whether the server serves the snapshots or leaves them to the workers."""
    serve = WorkerPool._serve

    def serve_then_pause(self, requests):
        serve(self, requests)
        time.sleep(0.008)  # both workers' next requests arrive before the next drain: mixed batches

    monkeypatch.setattr(WorkerPool, "_serve", serve_then_pause)
    kw = dict(workers=2, self_play_prob=0.3, opp_weights={"lookahead": 0.05, "random": 0.05, "snapshot": 0.6})
    runs = {}
    for name, mode, serve_snapshots in (("off", "off", True), ("on", "on", True), ("policy", "on", False)):
        tr = PPOTrainer(tiny(tmp_path / name, inference_server=mode, serve_snapshots=serve_snapshots, **kw))
        assert tr.inference_server is (mode == "on")
        tr._add_snapshot(tr.run_dir / "s.pt")
        try:
            runs[name] = [tr.collect() for _ in range(2)]  # the second collect continues running games
        finally:
            tr.close()
    for (t_off, s_off), (t_on, s_on), (t_pol, s_pol) in zip(runs["off"], runs["on"], runs["policy"]):
        assert "server" not in s_off
        for trajs, stats in ((t_on, s_on), (t_pol, s_pol)):
            _same_trajectories(t_off, trajs, logp_atol=1e-5)
            assert stats["transitions"] == s_off["transitions"] and stats["time"]["infer"] > 0
            srv = stats["server"]
            assert 0 < srv["forwards"] < srv["requests"], "some forwards batched requests of both workers"
        srv = s_on["server"]
        assert srv["rows_by_key"].get("snap_00000", 0) > 0 and srv["snapshot_forwards"] > 0, "snapshots served"
        assert srv["rows"] == s_on["infer_rows"] and srv["requests"] == s_on["infer_calls"]
        srv = s_pol["server"]
        assert set(srv["rows_by_key"]) == {POLICY_KEY} and srv["snapshot_forwards"] == 0  # snapshots stayed local
        assert srv["rows"] < s_pol["infer_rows"] and srv["requests"] < s_pol["infer_calls"]


def test_server_stats_are_logged_and_evicted_snapshots_stay_served(tmp_path, monkeypatch):
    """3 updates, 1 snapshot kept: running games against an evicted snapshot are still served."""
    seen = []
    collect = PPOTrainer.collect

    def spy(self):
        out = collect(self)
        seen.append(({n for n, _, _ in self.snapshots}, out[1]["server"]))
        return out

    monkeypatch.setattr(PPOTrainer, "collect", spy)
    tr = PPOTrainer(tiny(tmp_path, workers=2, inference_server="on", serve_snapshots=True, total_updates=3, max_snapshots=1,
                         self_play_prob=0.2, opp_weights={"random": 0.2, "snapshot": 0.8}))
    tr.train(log=lambda *_: None)
    rows = [json.loads(line) for line in (tr.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    for row in rows:
        assert row["server_forward_s"] > 0 and row["server_forwards"] > 0 and row["server_rows_mean"] >= 1
        assert row["server_requests"] >= row["server_forwards"] and row["server_busy_s"] >= row["server_forward_s"]
        assert row["server_requests"] >= row["server_drains"] > 0
        assert row["server_snapshot_forward_s"] <= row["server_forward_s"]
        assert row["server_snapshot_forwards"] <= row["server_forwards"]
        assert "wt_infer_frac" in row
    assert json.loads((tr.run_dir / "config.json").read_text(encoding="utf-8"))["inference_server_active"] is True
    assert rows[0]["server_snapshot_forwards"] == 0 and rows[1]["server_snapshot_forwards"] > 0  # pool from update 1
    evicted = [set(srv["rows_by_key"]) - {POLICY_KEY} - pool for pool, srv in seen]
    assert any(evicted), "a game against a snapshot that left the pool continued under the server"


def test_server_mode_worker_death_while_serving(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    server = InferenceServer(tr.net, "cpu", tr.n_actions)
    pool = WorkerPool([worker_init(tr, w, 2) for w in range(2)], in_process=False, timeout=60, server=server)
    procs = list(pool.procs)
    serve, calls = server.serve, []

    def serve_then_kill(requests):
        calls.append(len(requests))
        if len(calls) == 20:
            procs[0].kill()  # e.g. the OOM killer while the learner is serving
        return serve(requests)

    server.serve = serve_then_kill
    try:
        server.begin([])
        _start_collecting(pool, None, (10 ** 9, 10 ** 9))
        with pytest.raises(RolloutWorkerError, match=rf"rollout worker 0 .*exit code {KILLED}\b"):
            pool._gather()
        assert len(calls) >= 20 and not any(p.is_alive() for p in procs)
    finally:
        pool.close(force=True)


def test_server_mode_worker_errors_and_timeouts_stop_every_worker(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))

    def make_pool():
        server = InferenceServer(tr.net, "cpu", tr.n_actions)
        return WorkerPool([worker_init(tr, w, 2) for w in range(2)], in_process=False, timeout=60, server=server)

    pool = make_pool()  # a malformed reply makes the worker fail: its traceback reaches the learner
    procs = list(pool.procs)
    pool._serve = lambda requests: [pool.conns[i].send(("bogus",)) for i, _ in requests]
    try:
        with pytest.raises(RolloutWorkerError, match=r"(?s)rollout worker [01] failed.*expected logits"):
            pool.collect(0, None, [], 10 ** 9)
        assert not any(p.is_alive() for p in procs)
    finally:
        pool.close(force=True)
    pool = make_pool()  # requests that are never answered: the timeout stops everything
    procs = list(pool.procs)
    try:
        pool.collect(0, None, [], 50)  # workers are up
        dropped = []
        pool._serve = dropped.append
        pool.timeout = 1.5
        with pytest.raises(RolloutWorkerError, match="timed out"):
            pool.collect(1, None, [], 10 ** 9)
        assert dropped and not any(p.is_alive() for p in procs)
    finally:
        pool.close(force=True)


@pytest.mark.filterwarnings("ignore:the training deal seeds")  # batch_steps 1e8: the projection warns
def test_interrupted_serving_stops_busy_workers_at_once(tmp_path, monkeypatch):
    tr = PPOTrainer(tiny(tmp_path, workers=1, batch_steps=10 ** 8, inference_server="on"))
    seen = {}

    def interrupted(self, requests):
        seen.update(t=time.time(), procs=list(tr.pool.procs))
        raise KeyboardInterrupt  # Ctrl-C while the learner runs a forward for the workers

    monkeypatch.setattr(InferenceServer, "serve", interrupted)
    with pytest.raises(KeyboardInterrupt):
        tr.train(log=lambda *_: None)
    assert time.time() - seen["t"] < 3.0
    assert tr.pool is None and not any(p.is_alive() for p in seen["procs"])


@pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="no GPU device")
def test_inference_server_on_the_gpu_device(tmp_path):
    dev = "cuda" if torch.cuda.is_available() else "mps"
    tr = PPOTrainer(tiny(tmp_path, workers=2, device=dev, inference_server="on"))
    try:
        trajs, stats = tr.collect()
    finally:
        tr.close()
    assert stats["server"]["forwards"] > 0 and trajs
    for t in trajs:  # the stored behaviour log-probs are the device policy's (checked on the CPU copy)
        mask = _unpack(tr, t)
        assert mask[np.arange(len(t["act"])), t["act"]].all()
        with torch.no_grad():
            lp = torch.log_softmax(tr.actor.policy_logits(torch.from_numpy(t["obs"].astype(np.float32)),
                                                          torch.from_numpy(mask)), -1)
        want = lp.gather(1, torch.from_numpy(t["act"].astype(np.int64))[:, None]).squeeze(1).numpy()
        np.testing.assert_allclose(t["logp"], want, rtol=0, atol=1e-3)
