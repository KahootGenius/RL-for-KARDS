"""Regression tests for the Stage 3 pipeline review: bf16 autocast on CUDA, the best.pt selection,
the training seed budget, belief metrics, train.py refusals and argument parsing, the pipe format of
the rollout workers, scenario validity and the documented defaults.

Tiny configs on the CPU, like tests/test_ppo.py.
"""
from __future__ import annotations

import importlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as tF
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_map

from cardgame.cards import load_ruleset
from cardgame.rl import ppo
from cardgame.rl.ppo import (BEST_MIN_CELL, SELECTION_SEED_BASE, TRAIN_SEED_BASE, TRAIN_SEEDS_PER_RUN, PPOConfig,
                             PPOTrainer, SeedBlockExhausted, best_score, eligible_probability)
from cardgame.rl.rollout import InferenceServer, SparseRows, WorkerInit, to_dense, to_sparse

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_ruleset()


def tiny(tmp_path, **kw) -> PPOConfig:
    base = dict(run_dir=str(tmp_path / "run"), workers=0, envs_per_worker=8, batch_steps=256, minibatch=128,
                micro_batch=64, epochs=2, d_model=32, layers=1, heads=2, ff=64, id_dim=4, ctx_dim=32, pair_dim=8,
                hidden=(32,), snapshot_every=1, max_snapshots=2, eval_every=0, device="cpu", torch_threads=1,
                tensorboard=False, select_games=0)
    base.update(kw)
    return PPOConfig(**base)


# ---------------------------------------------------------------- bf16 autocast (CUDA default amp=True)
# Ops CUDA autocast runs in fp32 (torch's autocast op reference: layer_norm, sum, softmax, exp, ...). CPU
# autocast keeps most of them in bf16, so the CUDA-only dtype mixes never show on a CPU. This mode
# emulates CUDA's policy inside torch.autocast("cpu"): those ops get fp32 inputs and give fp32 outputs.
_CUDA_FP32_OPS = {tF.layer_norm, torch.layer_norm, torch.sum, torch.Tensor.sum, torch.softmax, torch.log_softmax,
                  torch.Tensor.softmax, torch.Tensor.log_softmax, tF.softmax, tF.log_softmax, torch.exp,
                  torch.Tensor.exp, torch.cumsum, torch.Tensor.cumsum, torch.pow, torch.Tensor.pow,
                  torch.Tensor.__pow__, tF.binary_cross_entropy_with_logits, torch.log, torch.Tensor.log,
                  torch.prod, torch.Tensor.prod, torch.rsqrt, torch.Tensor.rsqrt, torch.reciprocal, torch.norm,
                  torch.Tensor.norm, tF.normalize}


class _CudaLikeAutocast(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _CUDA_FP32_OPS and torch.is_autocast_enabled("cpu"):
            def up(t):
                return t.float() if isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16 else t
            with torch.autocast("cpu", enabled=False):
                return func(*tree_map(up, args), **tree_map(up, kwargs))
        return func(*args, **kwargs)


AMP_CASES = [("transformer", {}), ("transformer", {"privileged_critic": True}), ("transformer", {"shared_trunk": True}),
             ("pooled", {}), ("pooled", {"privileged_critic": True}), ("mlp", {"privileged_critic": True})]


@pytest.mark.parametrize("arch,kw", AMP_CASES)
def test_learn_runs_under_cuda_like_bf16_autocast(tmp_path, arch, kw):
    """amp=True on CUDA (the desktop default) runs _values and _minibatch_backward under bf16 autocast, where
    the encoder's final LayerNorm is fp32 and the token embedding bf16: the scatter back must not mix them."""
    tr = PPOTrainer(tiny(tmp_path, arch=arch, epochs=1, **kw))
    trajs, _ = tr.collect()
    tr.amp = True  # the trainer turns amp on for CUDA only; _autocast follows self.device (cpu here)
    seen = []
    if arch == "transformer":
        enc = tr.net.value_tower.encoder
        enc.register_forward_hook(lambda m, inp, out: seen.append((inp[0].dtype, out.dtype)))
    with _CudaLikeAutocast():
        out = tr.learn(trajs)
    assert all(np.isfinite(v) for v in out.values()) and out["kl"] < 0.05
    if arch == "transformer":  # the emulation reproduces CUDA's mix: bf16 tokens in, fp32 LayerNorm out
        assert seen and all(o == torch.float32 and i == torch.bfloat16 for i, o in seen)


@pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="no GPU device")
@pytest.mark.parametrize("arch,kw", AMP_CASES[:2] + AMP_CASES[3:4])
def test_learn_runs_under_device_bf16_autocast(tmp_path, arch, kw):
    dev = "cuda" if torch.cuda.is_available() else "mps"
    tr = PPOTrainer(tiny(tmp_path, arch=arch, epochs=1, device=dev, **kw))
    trajs, _ = tr.collect()
    tr.amp = True
    out = tr.learn(trajs)
    assert all(np.isfinite(v) for v in out.values()) and out["kl"] < 0.05 and out["clipfrac"] < 0.05


# ---------------------------------------------------------------- best.pt: quick-eval cells and selection
def test_quick_eval_min_cell_needs_every_fixed_cell(tmp_path):
    """eval_deals 20 plays only 10 fixed deals, which never reach the cells 2-2, 2-3 and 3-3: no min_cell,
    so the eval cannot make best.pt eligible."""
    tr = PPOTrainer(tiny(tmp_path))
    n = CONFIG.n_decks
    part = tr.quick_eval(20)
    assert len(part["cells"]) == 7 < n * (n + 1) // 2 and part["min_cell"] is None
    assert best_score(dict(part, min_cell=None))[0] == 0 and eligible_probability(part) == 0.0
    full = tr.quick_eval(2 * n * n)
    assert len(full["cells"]) == n * (n + 1) // 2 and full["min_cell"] == min(full["cells"].values())
    assert {k: c[0] / c[1] for k, c in full["cell_counts"].items()} == full["cells"]
    assert sum(c[1] for c in full["cell_counts"].values()) == full["fixed_games"]


def test_eligible_probability_is_the_posterior_that_every_cell_clears_the_bar():
    def ev(*cells):
        return {"min_cell": min(w / g for w, g in cells), "cell_counts": {str(i): list(c) for i, c in enumerate(cells)}}

    # P(p >= 0.6 | w of g) with a uniform prior = P(Binomial(g + 1, 0.6) <= w), cross-checked numerically
    grid = np.linspace(0, 1, 200001)
    for w, g in ((16, 32), (20, 32), (10, 16), (0, 2), (32, 32)):
        dens = grid ** w * (1 - grid) ** (g - w)
        want = dens[grid >= BEST_MIN_CELL].sum() / dens.sum()
        assert eligible_probability(ev((w, g))) == pytest.approx(want, abs=1e-4)
    assert eligible_probability(ev((20, 32), (20, 32))) == pytest.approx(eligible_probability(ev((20, 32))) ** 2)
    assert eligible_probability({"min_cell": None, "cell_counts": {"0-0": [5, 5]}}) == 0.0
    # one 19/32 cell (point estimate 0.59 < 0.6) makes an eval ineligible under the rule, yet it is still
    # roughly a coin flip that the cell clears 0.6: the second candidate ranking keeps such evals
    dipped = ev(*[(22, 32)] * 9 + [(19, 32)])
    assert best_score(dict(dipped, win_rate=0.7))[0] == 0 and 0.05 < eligible_probability(dipped) < 0.5


def _fake_ev(random_rate, cells_rate, games=32):
    n = CONFIG.n_decks
    keys = [f"{i}-{j}" for i in range(n) for j in range(i, n)]
    rates = dict(zip(keys, cells_rate)) if isinstance(cells_rate, (list, tuple)) else dict.fromkeys(keys, cells_rate)
    counts = {k: [round(r * games), games] for k, r in rates.items()}
    rates = {k: c[0] / c[1] for k, c in counts.items()}
    return {"win_rate": (random_rate + float(np.mean(list(rates.values())))) / 2, "draw_rate": 0.0, "games": 512,
            "random_win_rate": random_rate, "random_draw_rate": 0.0, "random_games": 256,
            "fixed_win_rate": float(np.mean(list(rates.values()))), "fixed_games": 256,
            "min_cell": min(rates.values()), "cells": rates, "cell_counts": counts}


def test_selection_pass_overrides_a_lucky_quick_eval(tmp_path, monkeypatch):
    """Checkpoint A has a 0.5 cell, but in its quick evals every 32-game cell sample cleared 0.6, several
    barely (eligible; the best random-deck rate); B is strong everywhere, but one cell sample dipped to
    19/32. The quick-eval rule picks A; P(every cell >= 0.6) ranks B's evals first, so both reach the
    selection pass, which (with the verdict's sample sizes) finds A ineligible and picks B."""
    a_lucky = _fake_ev(0.76, [x / 32 for x in (26, 26, 24, 22, 22, 21, 21, 21, 20, 20)])
    b_unlucky = _fake_ev(0.71, [25 / 32] * 9 + [19 / 32])
    assert best_score(a_lucky)[0] == 1 and best_score(b_unlucky)[0] == 0
    assert eligible_probability(b_unlucky) > 2 * eligible_probability(a_lucky)
    quick = iter([b_unlucky, a_lucky, b_unlucky, a_lucky])
    truth = {"A": _fake_ev(0.75, [0.80] * 9 + [0.50], games=200), "B": _fake_ev(0.70, 0.69, games=200)}

    monkeypatch.setattr(PPOTrainer, "quick_eval", lambda self, n, deterministic=False: next(quick))
    seen = []

    def fake_selection(self, path, evaluator):
        update = torch.load(path, weights_only=False)["update"]
        seen.append(update)
        return truth["A" if update % 2 == 0 else "B"]

    monkeypatch.setattr(PPOTrainer, "selection_eval", fake_selection)
    tr = PPOTrainer(tiny(tmp_path, total_updates=4, eval_every=1, batch_steps=64, minibatch=64, select_top=1,
                         select_games=2000))
    logs = []
    tr.train(log=logs.append)
    run = tr.run_dir
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r.get("best_update") for r in rows] == [1, 2, None, None]  # the quick-eval pick: A (update 2)
    sel = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    assert sorted(seen) == [3, 4] and sel["chosen_update"] == 3 and sel["eligible"] is True  # B, newest of its kind
    assert sel["seed_base"] == SELECTION_SEED_BASE == 500_000_000 and sel["games_per_mode"] == 2000
    assert torch.load(run / "best.pt", weights_only=False)["update"] == 3
    assert sorted(p.name for p in run.glob("cand_*.pt")) == ["cand_00003.pt", "cand_00004.pt"]
    assert any("best.pt = cand_00003.pt" in line for line in logs)
    state = torch.load(run / "latest.pt", weights_only=False)
    assert [c["update"] for c in state["candidates"]] == [3, 4]


def test_selection_pass_end_to_end_and_resume(tmp_path, monkeypatch):
    """Real quick evals and a real selection pass (tiny sizes), then a resume of the finished run: the
    candidates come back from latest.pt and only the selection pass runs again."""
    tr = PPOTrainer(tiny(tmp_path, total_updates=2, eval_every=1, eval_deals=2, select_games=2, select_top=1,
                         batch_steps=64, minibatch=64))
    tr.train(log=lambda *_: None)
    run = tr.run_dir
    sel = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    assert 1 <= len(sel["candidates"]) <= 2
    n = CONFIG.n_decks
    for c in sel["candidates"]:
        s = c["select"]
        assert (s["random_games"], s["fixed_games"]) == (2, 2 * n * n) and len(s["cells"]) == n * (n + 1) // 2
        assert tuple(c["select_score"]) == best_score(s)
    chosen = torch.load(run / "best.pt", weights_only=False)
    assert chosen["update"] == sel["chosen_update"]
    want = torch.load(run / sel["chosen"], weights_only=False)["model"]
    assert all(torch.equal(v, want[k]) for k, v in chosen["model"].items())
    tr2 = PPOTrainer(tiny(tmp_path, total_updates=2, eval_every=1, eval_deals=2, select_games=2, select_top=1,
                          batch_steps=64, minibatch=64))
    tr2.resume(str(run / "latest.pt"))
    assert [c["update"] for c in tr2.candidates] == [c["update"] for c in sel["candidates"]]
    collected = []
    monkeypatch.setattr(PPOTrainer, "collect", lambda self: collected.append(1))
    logs = []
    tr2.train(log=logs.append)
    assert not collected and any(line.startswith("selection: best.pt =") for line in logs)


# ---------------------------------------------------------------- training deal seeds
def test_seed_block_exhaustion_stops_before_a_collect(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    tr.worker_games = [TRAIN_SEEDS_PER_RUN - 5]  # 5 deals left, 8 running games need more
    with pytest.raises(SeedBlockExhausted, match=r"--seed 0 are used up.*--resume .*latest\.pt --seed"):
        tr.collect()
    assert tr.pool is None  # nothing started: no worker fails mid-collect
    tr.worker_games = [0]
    trajs, _ = tr.collect()
    assert trajs and tr._games_per_collect >= 1
    b = tr.seed_budget()
    assert (b["start"], b["end"]) == (TRAIN_SEED_BASE, TRAIN_SEED_BASE + TRAIN_SEEDS_PER_RUN)
    assert b["used"] == tr.worker_games[0] and b["used"] + b["remaining"] == TRAIN_SEEDS_PER_RUN


def _finished_run(tmp_path, **kw) -> Path:
    tr = PPOTrainer(tiny(tmp_path, total_updates=1, batch_steps=64, minibatch=64, **kw))
    tr.train(log=lambda *_: None)
    return tr.run_dir / "latest.pt"


def test_resume_of_an_exhausted_seed_block(tmp_path):
    latest = _finished_run(tmp_path)
    state = torch.load(latest, weights_only=False)
    state["worker_games"] = [TRAIN_SEEDS_PER_RUN]  # every seed of the block dealt
    torch.save(state, latest)
    with pytest.raises(SeedBlockExhausted, match=r"used up.*--seed <another seed"):
        PPOTrainer(tiny(tmp_path, total_updates=2)).resume(str(latest))
    tr = PPOTrainer(tiny(tmp_path, total_updates=2, seed=7))  # an explicit other --seed: that seed's block
    tr.resume(str(latest))
    assert tr.seed_base == TRAIN_SEED_BASE + 7 * TRAIN_SEEDS_PER_RUN and tr.update == 1
    assert tr.seed_budget()["remaining"] == TRAIN_SEEDS_PER_RUN


def _run_main(monkeypatch, capsys, argv):
    train = importlib.import_module("train")
    with pytest.raises(SystemExit) as exc:
        train.main(argv)
    err = capsys.readouterr().err.strip()
    return exc.value.code, err


def test_train_cli_refusals_are_one_line_with_exit_code_2(tmp_path, monkeypatch, capsys):
    latest = _finished_run(tmp_path)
    common = ["--resume", str(latest), "--updates", "2", "--workers", "0", "--device", "cpu"]
    code, err = _run_main(monkeypatch, capsys, common + ["--layers", "2"])  # the architecture is the checkpoint's
    assert code == 2 and len(err.splitlines()) == 1 and err.startswith("train.py: error: cannot resume from")
    assert "layers (checkpoint 1, config 2, --layers)" in err and "Traceback" not in err
    code, err = _run_main(monkeypatch, capsys, common + ["--no-belief", "--privileged-critic"])
    assert code == 2 and "belief" in err and "--no-belief" in err and "--privileged-critic" in err
    state = torch.load(latest, weights_only=False)
    state["worker_games"] = [TRAIN_SEEDS_PER_RUN]
    torch.save(state, latest)
    code, err = _run_main(monkeypatch, capsys, common)
    assert code == 2 and len(err.splitlines()) == 1 and "used up" in err and "--seed" in err
    code, err = _run_main(monkeypatch, capsys, ["--run-dir", str(tmp_path / "x"), "--seed", "200"])
    assert code == 2 and err.startswith("train.py: error: invalid settings: seed must be in [0, 100)")
    train = importlib.import_module("train")

    def missing_data():
        raise FileNotFoundError(2, "No such file or directory", str(ROOT / "cardgame" / "data" / "cards.json"))

    monkeypatch.setattr(train, "load_ruleset", missing_data)
    code, err = _run_main(monkeypatch, capsys, ["--run-dir", str(tmp_path / "y"), "--workers", "0"])
    assert code == 2 and "resume" not in err and "data file is missing" in err and "cards.json" in err


def test_seed_block_exhaustion_mid_run_is_a_one_line_stop(tmp_path, monkeypatch, capsys):
    latest = _finished_run(tmp_path)

    def exhausted(self):
        raise SeedBlockExhausted(self._exhausted_message("test"))

    monkeypatch.setattr(PPOTrainer, "_check_seed_budget", exhausted)
    code, err = _run_main(monkeypatch, capsys, ["--resume", str(latest), "--updates", "3", "--workers", "0"])
    assert code == 2 and len(err.splitlines()) == 1
    assert err.startswith("train.py: error: training stopped after 1 completed updates") and "--seed" in err


def test_opp_weights_accept_powershell_style_separate_tokens():
    """PowerShell passes an unquoted lookahead=0.3,snapshot=0.2 as two arguments."""
    train = importlib.import_module("train")
    parser = train.build_parser()
    for argv in (["--opp-weights", "lookahead=0.3,snapshot=0.2"], ["--opp-weights", "lookahead=0.3", "snapshot=0.2"],
                 ["--opp-weights", "lookahead=0.3", "snapshot=0.2", "--updates", "2"]):
        cfg = train.make_config(parser.parse_args(argv))
        assert cfg.opp_weights == {"lookahead": 0.3, "random": 0.05, "snapshot": 0.2}, argv
    with pytest.raises(SystemExit):
        parser.parse_args(["--opp-weights", "lookahead=0.3", "snapshot"])


# ---------------------------------------------------------------- belief metrics are epoch-0 data
def test_belief_loss_is_logged_from_epoch_zero(tmp_path, monkeypatch):
    tr = PPOTrainer(tiny(tmp_path, epochs=4, target_kl=0.0, belief_coef=1.0, lr=3e-3))
    trajs, _ = tr.collect()
    seen = []
    orig = PPOTrainer._minibatch_backward

    def spy(self, data, idx, belief_stats=False):
        out = orig(self, data, idx, belief_stats)
        seen.append((belief_stats, out["belief_loss"], out.get("belief_acc")))
        return out

    monkeypatch.setattr(PPOTrainer, "_minibatch_backward", spy)
    out = tr.learn(trajs)
    epoch0 = [loss for first, loss, _ in seen if first]
    assert len(seen) == 4 * len(epoch0)
    assert out["belief_loss"] == pytest.approx(np.mean(epoch0))
    assert out["belief_loss_all"] == pytest.approx(np.mean([loss for _, loss, _ in seen]))
    assert out["belief_acc"] == pytest.approx(np.mean([acc for first, _, acc in seen if first]))
    assert out["belief_loss_all"] < out["belief_loss"]  # later epochs fit the same batch


# ---------------------------------------------------------------- pipes: sparse encodings, I/O time
def test_sparse_rows_round_trip_exactly_and_are_small(tmp_path):
    tr = PPOTrainer(tiny(tmp_path))
    trajs, _ = tr.collect()
    obs = np.concatenate([t["obs"] for t in trajs])
    obs[0, :3] = np.array([-0.0, 65504, 1e-7], dtype=np.float16)  # signed zero, the largest and a subnormal
    sp = to_sparse(obs)
    assert isinstance(sp, SparseRows) and sp.indices.dtype == np.uint16 and sp.values.dtype == np.float16
    back = to_dense(sp)
    assert back.dtype == np.float16 and back.view(np.uint16).tobytes() == obs.view(np.uint16).tobytes()  # bit-exact
    assert len(pickle.dumps(sp)) * 5 < len(pickle.dumps(obs))  # ~10x smaller (about 4% of the entries are nonzero)
    out = np.zeros_like(obs)
    assert to_dense(sp, out=out) is out and np.array_equal(out.view(np.uint16), obs.view(np.uint16))
    assert to_dense(obs) is obs and to_sparse(obs[:0]).shape == (0, obs.shape[1])


def test_spawned_workers_return_dense_obs_and_the_pipe_time_is_logged(tmp_path):
    tr = PPOTrainer(tiny(tmp_path, workers=2, total_updates=1, inference_server="on"))
    tr.train(log=lambda *_: None)
    row = json.loads((tr.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["pool_io_s"] > 0 and row["server_busy_s"] > 0
    tr = PPOTrainer(tiny(tmp_path / "local", workers=2))
    try:
        trajs, stats = tr.collect()
    finally:
        tr.close()
    assert stats["io_s"] > 0 and trajs
    assert all(isinstance(t["obs"], np.ndarray) and t["obs"].dtype == np.float16 for t in trajs)


def test_snapshot_serving_is_off_unless_asked():
    """SPEC 8 / 8.1: serve_snapshots defaults to False everywhere (config, worker init, server)."""
    import dataclasses
    import inspect
    assert PPOConfig().serve_snapshots is False
    assert {f.name: f.default for f in dataclasses.fields(WorkerInit)}["serve_snapshots"] is False
    assert inspect.signature(InferenceServer).parameters["serve_snapshots"].default is False
    spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")
    assert "`serve_snapshots`\n  (False; §8.1)" in spec or "`serve_snapshots` (False; §8.1)" in spec
    assert "(True; §8.1)" not in spec


# ---------------------------------------------------------------- scenarios
def test_scenarios_reject_cards_with_random_effects():
    from cardgame.scenarios import SCENARIOS, build_position, check_position, rng_dependent_cards, scenario_config
    from cardgame.scenarios import U, Scenario, kills
    cfg = scenario_config(CONFIG)
    ids = sorted(cfg.cards.cards[i].id for i in rng_dependent_cards(cfg))
    assert ids == ["infiltrator", "shrapnel", "skirmisher", "sniper"]
    for sc in SCENARIOS:
        assert check_position(sc, sc.build(cfg)) == [], sc.name
    # Shrapnel (2 x 1 random damage) against three 1-hp units: solvable only under some hidden RNG states
    volley = next(n for n in cfg.deck_names if "shrapnel" in {cfg.cards.cards[c].id for c in cfg.decks[
        cfg.deck_names.index(n)]})
    sc = Scenario("rng_probe", ("test",), (volley, volley),
                  lambda config=None: build_position(config, decks=(volley, volley), round=4, my_hand=["shrapnel"],
                                                     opp_back=[U("longbowman", hp=1)] * 3),
                  kills("longbowman"))
    problems = check_position(sc, sc.build(cfg))
    assert any("random effects" in p and "shrapnel" in p for p in problems)


def test_mid_game_scenarios_show_the_agent_a_real_deck():
    """The agent's deck holds the rest of its decklist (deck token and deck size as in training); the
    opponent's deck stays empty, so the survival checks see no hidden draw."""
    from cardgame.scenarios import SCENARIOS, agent_deck_rest, scenario_config
    cfg = scenario_config(CONFIG)
    cards = cfg.cards.cards
    for sc in SCENARIOS:
        g = sc.build(cfg)
        if sc.name == "mulligan_sanity":
            continue
        p = g.current
        obs = g.observe(p)
        own_board = [u for u in list(g.backline[p]) + list(g.frontline) if u.owner == p]
        expect = cfg.deck_size - sum(not cards[c].token for c in g.hands[p]) - sum(
            not cards[u.card].token for u in own_board)
        assert obs.my_deck_size == expect == len(agent_deck_rest(g, p)) > 25, sc.name
        assert sum(obs.my_deck_counts) == obs.my_deck_size and obs.opp_deck_size == 0, sc.name


def test_readme_train_commands_parse():
    """Every `python train.py ...` command in the README runs with the current flags."""
    train = importlib.import_module("train")
    parser = train.build_parser()
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    cmds = re.findall(r"^\s*python train\.py ([^\n#`]*)", text, flags=re.M)
    assert len(cmds) >= 4
    for cmd in cmds:
        parser.parse_args(cmd.replace("\\", "").split())
    assert re.search(r"python eval\.py --agent \S+ --baseline \S+", text), "the acceptance command measures (3)"
