"""PPO learner + rollout workers (Stage 2): bookkeeping, seeding, protocol, checkpoints, CLI."""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from cardgame.cards import load_ruleset
from cardgame.engine import DRAW, Game
from cardgame.rl import ppo
from cardgame.rl.ppo import (QUICK_EVAL_SEED_BASE, TRAIN_SEED_BASE, TRAIN_SEEDS_PER_RUN, PPOConfig, PPOTrainer,
                             select_device)
from cardgame.rl.rollout import RolloutWorker, RolloutWorkerError, WorkerInit, WorkerPool, _Env, _Opponent

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_ruleset()


def tiny(tmp_path, **kw) -> PPOConfig:
    base = dict(run_dir=str(tmp_path / "run"), workers=0, envs_per_worker=8, batch_steps=256, minibatch=128,
                epochs=2, d_model=16, id_dim=4, ctx_dim=32, pair_dim=16, snapshot_every=1, max_snapshots=2,
                eval_every=0, device="cpu", torch_threads=1, tensorboard=False)
    base.update(kw)
    return PPOConfig(**base)


def worker_init(tr: PPOTrainer, worker_id: int = 0, n_workers: int = 1) -> WorkerInit:
    cfg = tr.cfg
    return WorkerInit(worker_id=worker_id, n_workers=n_workers, seed_base=tr.seed_base,
                      seed_limit=tr.seed_block + TRAIN_SEEDS_PER_RUN, config=CONFIG, net_spec=tr.net.spec(),
                      envs=cfg.envs_per_worker, self_play_prob=cfg.self_play_prob, opp_weights=dict(cfg.opp_weights),
                      snapshots_per_worker=cfg.snapshots_per_worker)


# ---------------------------------------------------------------- learner math
def test_gae_uses_recomputed_values(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, gamma=0.9, gae_lambda=0.8))
    vals = torch.tensor([0.1, -0.2, 0.3])
    tr.net.value = lambda obs: vals[:len(obs)]  # the learner's value net, not the worker's
    traj = {"obs": np.zeros((3, tr.encoder.dim), np.float16),
            "mask": np.packbits(np.ones((3, tr.n_actions), bool), axis=1),
            "act": np.zeros(3, np.int16), "logp": np.zeros(3, np.float32), "version": np.zeros(3, np.int32),
            "reward": 1.0}
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


def test_collect_and_learn_in_process(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    trajs, stats = tr.collect()
    n = sum(len(t["act"]) for t in trajs)
    assert n >= tr.cfg.batch_steps and stats["transitions"] == n
    for t in trajs:
        mask = np.unpackbits(t["mask"], axis=1, count=tr.n_actions).astype(bool)
        T = len(t["act"])
        assert t["obs"].dtype == np.float16 and t["obs"].shape == (T, tr.encoder.dim)
        assert mask[np.arange(T), t["act"]].all(), "recorded action was not legal"
        assert (mask.sum(1) > 1).all(), "forced END_TURN decisions are not recorded"
        assert (t["obs"][:, 0] == 1).all(), "every recorded state is the learner's own turn"
        assert t["reward"] in (-1.0, 0.0, 1.0)
    out = tr.learn(trajs)
    assert all(np.isfinite(v) for v in out.values())
    for k, v in tr.net.state_dict().items():  # the CPU actor is refreshed after every update
        assert torch.equal(v.cpu(), tr.actor.state_dict()[k])


def _bandit(tr, reward_a=1.0, reward_b=-1.0, n=64, logp_shift=(0.0, 0.0)):
    """n one-step trajectories taking legal action A (reward_a) and n taking B (reward_b) in one position.

    Behaviour log-probs are the current policy's, shifted by `logp_shift` (A, B) to set the PPO ratio."""
    g, rng = Game(CONFIG), random.Random(1)
    g.reset(5)
    while len(g.legal_actions()) < 3:
        la = g.legal_actions()
        g.step(la[rng.randrange(len(la))])
    m = g.legal_mask()
    x = tr.encoder.encode(g.observe(g.current_player()), m).astype(np.float16)
    a_act, b_act = (int(i) for i in np.flatnonzero(m)[1:3])
    xt, mt = torch.from_numpy(x.astype(np.float32))[None], torch.from_numpy(m)[None]
    with torch.no_grad():
        lp = torch.log_softmax(tr.net.policy_logits(xt, mt), -1)[0]
    trajs = []
    for act, r, shift in ((a_act, reward_a, logp_shift[0]), (b_act, reward_b, logp_shift[1])):
        trajs += [{"obs": x[None], "mask": np.packbits(m[None], axis=1), "act": np.array([act], np.int16),
                   "logp": np.array([float(lp[act]) + shift], np.float32), "version": np.zeros(1, np.int32),
                   "reward": r} for _ in range(n)]
    return trajs, xt, mt, a_act, b_act


def _zero_values(*trainers):
    for tr in trainers:  # value 0 everywhere: advantage = reward, and the value loss has no gradient
        tr.net.value = lambda obs: torch.zeros(len(obs))


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
    tr = PPOTrainer(tiny(tmp_path, vf_coef=0.0, ent_coef=0.0, epochs=1, minibatch=1024, target_kl=0.0))
    _zero_values(tr)
    trajs, *_ = _bandit(tr, 1.0, -1.0, logp_shift=(-1.0, 1.0))
    before = {k: v.clone() for k, v in tr.net.state_dict().items()}
    tr.learn(trajs)
    for k, v in tr.net.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_entropy_bonus_raises_the_entropy(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, vf_coef=0.0, ent_coef=0.5, epochs=4, minibatch=32, target_kl=0.0))
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
    tr = PPOTrainer(tiny(tmp_path, envs_per_worker=32, batch_steps=2000))
    _, stats = tr.collect()
    assert stats["env_steps"] > 4096 and tr.pool.local.init.parent_pid == 0


def test_random_opponent_has_its_own_rng_stream(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    worker = RolloutWorker(worker_init(tr))
    opp_rng = worker.scripted["random"].agent.rng
    assert [opp_rng.random() for _ in range(4)] != [worker.rng.random() for _ in range(4)]


def test_finished_games_reward_the_winner(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    worker = RolloutWorker(worker_init(tr))
    step = (np.zeros(tr.encoder.dim, np.float16), np.ones(tr.n_actions, bool), 0, 0.0, 0)
    for seats, opp in (((0,), worker.scripted["greedy"]), ((1,), worker.scripted["random"]),
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
            if opp.kind == "self":  # result logged from the first player's side
                assert results == [("self_first", g.deck_ids[g.first_player], g.deck_ids[1 - g.first_player],
                                    0.0 if winner == DRAW else 1.0 if winner == g.first_player else -1.0)]
            else:
                assert [r[3] for r in results] == expected and results[0][0] == opp.kind


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


def test_opponent_mix_follows_the_config(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    worker = RolloutWorker(worker_init(tr))
    worker.pool_names = worker.active = ["snap_a"]
    kinds = Counter(worker._sample_opponent()[0].kind for _ in range(20000))
    assert abs(kinds["self"] / 20000 - 0.5) < 0.02
    rest = 20000 - kinds["self"]
    for kind, w in (("greedy", 0.15), ("random", 0.05), ("snapshot", 0.30)):
        assert abs(kinds[kind] / rest - w / 0.5) < 0.03, kinds
    worker.active = []
    assert "snapshot" not in Counter(worker._sample_opponent()[0].kind for _ in range(2000))


def test_every_policy_only_observes_its_own_seat(tmp_path, monkeypatch):
    tr = PPOTrainer(tiny(tmp_path))
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


def test_snapshot_opponents_play_with_the_frozen_net(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    worker = RolloutWorker(dataclasses.replace(worker_init(tr), self_play_prob=0.0, opp_weights={"snapshot": 1.0}))
    worker.update_pool(add={"s": (tr.net.spec(), tr._weights())}, names=["s"])
    snap, calls = worker.snapshots["s"], []
    policy_logits = snap.policy_logits
    snap.policy_logits = lambda *a, **kw: (calls.append(len(a[0])), policy_logits(*a, **kw))[1]
    out = worker.collect(64)
    assert calls, "the snapshot net never chose an action"
    assert {r[0] for r in out["results"]} == {"snapshot"}


def test_spawned_workers_collect_and_report_errors(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, workers=2))
    try:
        trajs, stats = tr.collect()
        assert stats["transitions"] >= tr.cfg.batch_steps and trajs
        assert tr.worker_games[0] > 0 and tr.worker_games[1] > 0
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
        assert pool.conns[1].poll(60)
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
        with pytest.raises(RolloutWorkerError, match=r"rollout worker 1 .*exit code -9"):
            tr.collect()
        assert not any(p.is_alive() for p in procs)
    finally:
        tr.close(force=True)
    pool = WorkerPool([worker_init(tr, w, 2) for w in range(2)], in_process=False, timeout=60)
    procs = list(pool.procs)
    try:
        _start_collecting(pool, tr._weights(), (10 ** 9, 10 ** 9))
        procs[0].kill()  # dies mid-collection: found while waiting for the replies
        with pytest.raises(RolloutWorkerError, match=r"rollout worker 0 .*exit code -9"):
            pool._gather()
        assert not any(p.is_alive() for p in procs)
    finally:
        pool.close(force=True)


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


# ---------------------------------------------------------------- checkpoints, config, CLI
def test_checkpoint_resume(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, total_updates=3))
    tr.train(log=lambda *_: None)
    state = torch.load(tr.run_dir / "latest.pt", weights_only=False)
    assert state["update"] == 3 and [n for n, _ in state["snapshots"]] == ["snap_00002", "snap_00003"]
    tr2 = PPOTrainer(tiny(tmp_path, total_updates=5))
    tr2.resume(str(tr.run_dir / "latest.pt"))
    assert tr2.update == 3 and [n for n, _, _ in tr2.snapshots] == ["snap_00002", "snap_00003"]
    assert tr2.seed_base == state["seed_base"] + state["n_workers"] * (max(state["worker_games"]) + 1)
    assert tr2.lr_at(3) == pytest.approx(state["lr"])
    for k, v in tr.net.state_dict().items():
        assert torch.equal(v, tr2.net.state_dict()[k])
    with pytest.raises(ValueError, match="architecture"):
        PPOTrainer(tiny(tmp_path, d_model=8)).resume(str(tr.run_dir / "latest.pt"))


@pytest.mark.parametrize("bad", [{"seed": 100}, {"snapshot_every": 0}, {"envs_per_worker": 0}, {"minibatch": 0},
                                 {"eval_every": 1, "eval_deals": 0}, {"arch": "rnn"}, {"workers": -2},
                                 {"self_play_prob": 1.5}, {"opp_weights": {"greedy": 0.2, "grredy": 0.1}},
                                 {"opp_weights": {"greedy": 0.2, "random": -0.1}},
                                 {"opp_weights": {"greedy": float("nan")}},
                                 {"opp_weights": {"greedy": 0.0, "random": 0.0, "snapshot": 0.5}}])
def test_config_validation(tmp_path, bad):
    with pytest.raises(ValueError):
        PPOTrainer(tiny(tmp_path, **bad))


def test_opponent_weights_cli_merges_over_the_defaults(tmp_path):
    PPOTrainer(tiny(tmp_path, self_play_prob=1.0, opp_weights={"snapshot": 1.0}))  # pool never used: fine
    train = importlib.import_module("train")
    parser = train.build_parser()
    cfg = train.make_config(parser.parse_args(["--opp-weights", "greedy=0.4,snapshot=0"]))
    assert cfg.opp_weights == {"greedy": 0.4, "random": 0.05, "snapshot": 0.0}
    assert train.make_config(parser.parse_args([])).opp_weights == PPOConfig().opp_weights
    with pytest.raises(SystemExit):
        parser.parse_args(["--opp-weights", "greedy"])


def test_quick_eval_covers_every_deck_pair(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    res = tr.quick_eval(32)
    n = CONFIG.n_decks
    assert len(res["cells"]) == n * (n + 1) // 2
    assert 0 <= res["min_cell"] <= 1 and 0 <= res["win_rate"] <= 1


def test_train_cli_tensorboard_and_resume(tmp_path):
    pytest.importorskip("tensorboard")
    run = tmp_path / "cli"
    common = [sys.executable, "train.py", "--run-dir", str(run), "--workers", "0", "--envs-per-worker", "8",
              "--batch-steps", "128", "--minibatch", "64", "--d-model", "16", "--ctx-dim", "32", "--pair-dim", "16",
              "--eval-every", "0", "--device", "cpu", "--torch-threads", "1"]
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    subprocess.run(common + ["--updates", "2", "--ent-coef", "0.02", "--opp-weights", "greedy=0.3"], cwd=ROOT,
                   check=True, capture_output=True, env=env)
    assert any((run / "tb").iterdir()), "TensorBoard event file written"
    subprocess.run([sys.executable, "train.py", "--resume", str(run / "latest.pt"), "--updates", "3",
                    "--opp-weights", "random=0.1"], cwd=ROOT, check=True, capture_output=True, env=env)
    rows = [json.loads(line) for line in open(run / "metrics.jsonl", encoding="utf-8")]
    assert [r["update"] for r in rows] == [1, 2, 3]
    args = torch.load(run / "latest.pt", weights_only=False)["args"]
    assert args["ent_coef"] == 0.02 and args["batch_steps"] == 128 and args["total_updates"] == 3
    assert args["opp_weights"] == {"greedy": 0.3, "random": 0.1, "snapshot": 0.3}  # merged over the saved run's
