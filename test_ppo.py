"""PPO components: masked network, numpy inference, GAE, rollout bookkeeping, checkpoints, CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest
import torch

from conftest import CONFIG, ROOT
from cardgame.engine import Game
from cardgame.features import ObservationEncoder
from cardgame.rl.agent import PPOAgent
from cardgame.rl.network import NumpyPolicy, PolicyValueNet
from cardgame.rl.ppo import PPOConfig, PPOTrainer, TRAIN_SEED_BASE, QUICK_EVAL_SEED_BASE

ENC = ObservationEncoder(CONFIG)
N_ACTIONS = Game(CONFIG).num_actions


def tiny_cfg(tmp_path, **kw) -> PPOConfig:
    base = dict(run_dir=str(tmp_path / "run"), num_envs=8, batch_steps=256, minibatch=64, epochs=2,
                hidden=(32, 32), snapshot_every=1, max_snapshots=2, eval_every=0, torch_threads=1)
    base.update(kw)
    return PPOConfig(**base)


def sample_positions(n_games=20):
    g, rng, out = Game(CONFIG), np.random.default_rng(0), []
    for s in range(n_games):
        g.reset(s)
        while not g.done:
            out.append((ENC.encode(g.observe(g.current)), g.legal_mask()))
            la = g.legal_actions()
            g.step(la[rng.integers(len(la))])
    X = np.stack([x for x, _ in out])
    M = np.stack([m for _, m in out])
    return torch.from_numpy(X), torch.from_numpy(M)


# ---------------------------------------------------------------- network
def test_numpy_policy_matches_torch_logits():
    torch.manual_seed(0)
    net = PolicyValueNet(ENC.dim, N_ACTIONS, (64, 64))
    X, _ = sample_positions(5)
    with torch.no_grad():
        ref = net.policy(X).numpy()
    assert np.allclose(NumpyPolicy(net).logits(X.numpy()), ref, atol=1e-5)


def test_masked_actions_are_never_sampled_and_stats_are_finite():
    torch.manual_seed(0)
    net = PolicyValueNet(ENC.dim, N_ACTIONS, (64, 64))
    with torch.no_grad():  # make illegal actions attractive: masking must still exclude them
        net.policy[-1].bias.fill_(5.0)
    X, M = sample_positions(10)
    for _ in range(20):
        a, logp, v = net.act(X, M)
        assert M[torch.arange(len(a)), a].all()
        assert torch.isfinite(logp).all() and torch.isfinite(v).all()
    logp, ent, _ = net.evaluate(X, M, a)
    assert torch.isfinite(logp).all() and torch.isfinite(ent).all() and (ent >= -1e-6).all()
    single = M.sum(1) == 1  # forced moves: zero entropy, log-prob 0
    if single.any():
        assert torch.allclose(ent[single], torch.zeros(int(single.sum())), atol=1e-5)
    da, _, _ = net.act(X, M, deterministic=True)
    assert M[torch.arange(len(da)), da].all()


# ---------------------------------------------------------------- trainer internals
def test_gae_matches_hand_computation(tmp_path):
    tr = PPOTrainer(tiny_cfg(tmp_path, gamma=0.9, gae_lambda=0.8))
    vals = [0.1, -0.2, 0.3]
    traj = [(np.zeros(ENC.dim, np.float32), np.ones(N_ACTIONS, bool), 0, -1.0, v) for v in vals]
    out = tr._process(traj, reward=1.0)
    g, lam = 0.9, 0.8
    d2 = 1.0 - vals[2]
    d1 = g * vals[2] - vals[1]
    d0 = g * vals[1] - vals[0]
    a2 = d2
    a1 = d1 + g * lam * a2
    a0 = d0 + g * lam * a1
    assert np.allclose(out["adv"], [a0, a1, a2], atol=1e-6)
    assert np.allclose(out["ret"], np.array([a0, a1, a2]) + np.array(vals), atol=1e-6)


def test_collect_records_only_learner_decisions(tmp_path):
    tr = PPOTrainer(tiny_cfg(tmp_path))
    batch = tr.collect()
    n = sum(len(b["act"]) for b in batch)
    assert n >= tr.cfg.batch_steps and tr.learner_steps == n
    for b in batch:
        T = len(b["act"])
        assert b["obs"].shape == (T, ENC.dim) and b["mask"].shape == (T, N_ACTIONS)
        assert b["mask"][np.arange(T), b["act"]].all(), "recorded action was not legal"
        assert (b["mask"].sum(1) > 1).all(), "forced END_TURN decisions should not be recorded"
        assert (b["obs"][:, 0] == 1.0).all(), "every recorded state is the learner's own turn"
        seat0 = b["obs"][:, 2]
        assert (seat0 == seat0[0]).all(), "a trajectory never switches seats"
        assert np.allclose(b["ret"], b["adv"] + b["val"])


def test_learn_returns_finite_stats(tmp_path):
    tr = PPOTrainer(tiny_cfg(tmp_path))
    stats = tr.learn(tr.collect())
    assert all(np.isfinite(v) for v in stats.values())
    assert all(torch.isfinite(p).all() for p in tr.net.parameters())


def test_opponent_sampling_follows_the_brief(tmp_path):
    tr = PPOTrainer(tiny_cfg(tmp_path))
    kinds, seats = Counter(), Counter()
    slot = tr.envs[0]
    for _ in range(4000):
        tr._reset_env(slot)
        kinds[slot.opp.kind] += 1
        if slot.opp.kind != "self":
            seats[slot.learner_seats] += 1
    assert abs(kinds["self"] / 4000 - 0.5) < 0.04
    assert abs(kinds["random"] - kinds["greedy"]) < 0.15 * kinds["greedy"]  # pool drawn uniformly
    assert abs(seats[(0,)] - seats[(1,)]) < 0.15 * seats[(0,)]


def test_snapshot_opponents_play_legal_actions(tmp_path):
    tr = PPOTrainer(tiny_cfg(tmp_path))
    tr._add_snapshot(tr.run_dir / "ckpt_00000.pt")
    snap = tr.pool[-1]
    assert snap.kind == "snapshot" and snap.policy is not None
    assert all(p.data_ptr() != q.data_ptr() for p, q in zip(snap.net.parameters(), tr.net.parameters()))
    actions = [None] * len(tr.envs)
    for _ in range(30):
        idxs = [i for i, s in enumerate(tr.envs) if not s.game.done]
        tr._snapshot_act(snap.policy, idxs, actions)
        for i in idxs:
            assert tr.envs[i].game.is_legal(actions[i])
            tr.envs[i].game.step(actions[i])
            if tr.envs[i].game.done:
                tr._reset_env(tr.envs[i])


def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        PPOTrainer(tiny_cfg(tmp_path, seed=100))  # would reach the quick-eval seed range
    with pytest.raises(ValueError):
        PPOTrainer(tiny_cfg(tmp_path, snapshot_every=0))
    tr = PPOTrainer(tiny_cfg(tmp_path, seed=99))
    assert TRAIN_SEED_BASE <= tr.next_seed < QUICK_EVAL_SEED_BASE


def test_checkpoint_resume_round_trip(tmp_path):
    cfg = tiny_cfg(tmp_path, total_updates=3)
    tr = PPOTrainer(cfg)
    tr.train(log=lambda *_: None)
    latest = tr.run_dir / "latest.pt"
    state = torch.load(latest, weights_only=False)
    assert state["update"] == 3 and [n for n, _ in state["snapshots"]] == ["snap_00002", "snap_00003"]
    assert {"model", "net", "update", "env_steps", "args"} <= set(state)  # SPEC §7 format

    tr2 = PPOTrainer(tiny_cfg(tmp_path, total_updates=5))
    tr2.resume(str(latest))
    # in-flight games restart on fresh deals: seeds continue after the saved counter, never repeat
    assert tr2.update == 3 and tr2.next_seed == state["next_seed"] + tr2.cfg.num_envs
    assert [o.name for o in tr2.pool] == ["random", "greedy", "snap_00002", "snap_00003"]
    assert tr2.lr_at(3) == pytest.approx(state["lr"])
    assert tr2.lr_at(5) == pytest.approx(cfg.lr * cfg.lr_final_frac)
    for p, q in zip(tr.net.parameters(), tr2.net.parameters()):
        assert torch.equal(p, q)


def test_lr_schedule_is_linear_and_continuous(tmp_path):
    tr = PPOTrainer(tiny_cfg(tmp_path, total_updates=10, lr=1e-3, lr_final_frac=0.1))
    lrs = [tr.lr_at(u) for u in range(11)]
    assert lrs[0] == pytest.approx(1e-3) and lrs[10] == pytest.approx(1e-4)
    assert np.allclose(np.diff(lrs), np.diff(lrs)[0])


# ---------------------------------------------------------------- agent wrapper and CLI
def test_ppo_agent_from_checkpoint(tmp_path):
    tr = PPOTrainer(tiny_cfg(tmp_path))
    path = tmp_path / "a.pt"
    tr.checkpoint(path)
    for det in (False, True):
        agent = PPOAgent.from_checkpoint(str(path), CONFIG, deterministic=det, seed=1)
        g = Game(CONFIG)
        for s in range(3):
            g.reset(s)
            agent.reset(s)
            while not g.done:
                la = g.legal_actions()
                a = agent.act(g.observe(g.current), la)
                assert a in la
                g.step(a)


def test_ppo_agent_rejects_mismatched_encoder(tmp_path):
    net = PolicyValueNet(ENC.dim + 1, N_ACTIONS, (8,))
    with pytest.raises(ValueError):
        PPOAgent(net, CONFIG)


def test_train_cli_and_resume_keep_saved_hyperparameters(tmp_path):
    run = tmp_path / "cli"
    common = [sys.executable, "train.py", "--run-dir", str(run), "--num-envs", "8", "--batch-steps", "128",
              "--minibatch", "64", "--hidden", "16", "--eval-every", "0", "--torch-threads", "1"]
    subprocess.run(common + ["--updates", "2", "--ent-coef", "0.02"], cwd=ROOT, check=True, capture_output=True)
    subprocess.run([sys.executable, "train.py", "--resume", str(run / "latest.pt"), "--updates", "3"],
                   cwd=ROOT, check=True, capture_output=True)
    rows = [json.loads(line) for line in open(run / "metrics.jsonl")]
    assert [r["update"] for r in rows] == [1, 2, 3]
    args = torch.load(run / "latest.pt", weights_only=False)["args"]
    assert args["ent_coef"] == 0.02 and args["batch_steps"] == 128 and args["total_updates"] == 3


# ---------------------------------------------------------------- final-review regressions
def test_every_policy_in_training_only_observes_its_own_seat(tmp_path, monkeypatch):
    tr = PPOTrainer(tiny_cfg(tmp_path))
    tr._add_snapshot(tr.run_dir / "ckpt_00000.pt")
    snap = tr.pool[-1]
    calls = []
    orig = Game.observe

    def spy(self, player):
        calls.append((player, self.current))
        return orig(self, player)

    monkeypatch.setattr(Game, "observe", spy)
    actions = [None] * len(tr.envs)
    for _ in range(10):
        idxs = [i for i, s in enumerate(tr.envs) if not s.game.done]
        tr._snapshot_act(snap.policy, idxs, actions)
        for i in idxs:
            tr.envs[i].game.step(actions[i])
    tr.collect()
    tr.quick_eval(2)
    assert calls and all(p == cur for p, cur in calls), [c for c in calls if c[0] != c[1]][:3]


def test_quick_eval_accounting(tmp_path):
    tr = PPOTrainer(tiny_cfg(tmp_path))
    res = tr.quick_eval(3)
    wins = res["win_rate"] * 6
    assert abs(wins - round(wins)) < 1e-9
    assert res["seat0_win_rate"] * 3 + res["seat1_win_rate"] * 3 == pytest.approx(wins)


def test_resume_into_another_directory_keeps_snapshots_resolvable(tmp_path):
    tr = PPOTrainer(tiny_cfg(tmp_path, total_updates=2))
    tr.train(log=lambda *_: None)
    other = tmp_path / "elsewhere"
    tr2 = PPOTrainer(tiny_cfg(tmp_path, run_dir=str(other), total_updates=3))
    tr2.resume(str(tr.run_dir / "latest.pt"))
    tr2.train(log=lambda *_: None)
    tr3 = PPOTrainer(tiny_cfg(tmp_path, run_dir=str(other), total_updates=4))
    tr3.resume(str(other / "latest.pt"))  # snapshot refs written by tr2 must still resolve
    assert [o.name for o in tr3.pool if o.kind == "snapshot"] == ["snap_00002", "snap_00003"]


def test_resume_enforces_a_smaller_snapshot_cap(tmp_path):
    tr = PPOTrainer(tiny_cfg(tmp_path, total_updates=3, max_snapshots=3))
    tr.train(log=lambda *_: None)
    tr2 = PPOTrainer(tiny_cfg(tmp_path, total_updates=5, max_snapshots=1))
    tr2.resume(str(tr.run_dir / "latest.pt"))
    assert [o.name for o in tr2.pool if o.kind == "snapshot"] == ["snap_00003"]


def test_lr_given_on_resume_rescales_the_schedule(tmp_path):
    tr = PPOTrainer(tiny_cfg(tmp_path, total_updates=2, lr=1e-3))
    tr.train(log=lambda *_: None)
    saved = torch.load(tr.run_dir / "latest.pt", weights_only=False)["lr"]
    tr2 = PPOTrainer(tiny_cfg(tmp_path, total_updates=4, lr=1e-4))
    tr2.resume(str(tr.run_dir / "latest.pt"))
    assert tr2.lr_at(2) == pytest.approx(saved * 0.1)


def test_resume_from_a_plain_checkpoint_stays_in_the_runs_seed_block(tmp_path):
    tr = PPOTrainer(tiny_cfg(tmp_path, seed=99, total_updates=1))
    tr.train(log=lambda *_: None)
    tr2 = PPOTrainer(tiny_cfg(tmp_path, seed=99, total_updates=2))
    tr2.resume(str(tr.run_dir / "ckpt_00001.pt"))  # no next_seed stored
    assert tr2.seed_block <= tr2.next_seed < QUICK_EVAL_SEED_BASE


@pytest.mark.parametrize("bad", [{"num_envs": 0}, {"minibatch": 0}, {"batch_steps": 0}, {"epochs": 0},
                                 {"eval_every": 1, "eval_deals": 0}, {"max_snapshots": -1}])
def test_more_config_validation(tmp_path, bad):
    with pytest.raises(ValueError):
        PPOTrainer(tiny_cfg(tmp_path, **bad))


def test_resume_with_a_different_architecture_is_a_clear_error(tmp_path):
    tr = PPOTrainer(tiny_cfg(tmp_path, total_updates=1))
    tr.train(log=lambda *_: None)
    with pytest.raises(ValueError, match="hidden"):
        PPOTrainer(tiny_cfg(tmp_path, hidden=(16,))).resume(str(tr.run_dir / "latest.pt"))
