"""No hidden-information leaks: observe(p), its encoding and p's legal actions depend only on what p
may see (SPEC §5). Hidden: the opponent's hand contents (except the cards p knows about: returned or
added cards, `known_hand`) and deck choice, deck order and content, burned cards and the RNG. Public:
the board, pending choices (with their previews), the history counters and the bookkeeping counts."""
from __future__ import annotations

import copy
import random
from collections import Counter

import numpy as np
import pytest

from cardgame.cards import ARMOR_BIT
from cardgame.engine import DRAW, Game, Observation, PendingView, UnitView
from cardgame.features import ENCODER_VERSION, ObservationEncoder
from conftest import (CONFIG, DECK_PAIRS, N_CARDS, N_DECKS, PROFILE_PAIRS, H, WeightedPolicy, all_units,
                      check_invariants, iter_states, new_game, snapshot)

STAGE2_FIELDS = ("player", "is_my_turn", "went_first", "round", "my_deck", "my_coins", "opp_coins", "my_base_hp",
                 "opp_base_hp", "hand", "opp_hand_size", "my_deck_size", "opp_deck_size", "my_played",
                 "opp_played", "my_backline", "opp_backline", "frontline", "front_owner", "done", "result")
SPEC_FIELDS = STAGE2_FIELDS + (  # SPEC §5: Stage 3 fields are appended
    "phase", "mulligan_marks", "pending", "turn", "my_coin_bonus", "opp_coin_bonus", "my_decklist",
    "my_deck_counts", "my_known_hand", "opp_known_hand", "opp_revealed", "my_discard", "opp_discard",
    "my_graveyard", "opp_graveyard", "my_burned", "opp_burned", "my_history", "opp_history")
UNIT_VIEW_FIELDS = ("card", "atk", "hp", "max_hp", "armor", "defense", "nature", "move_cost", "summoned",
                    "moved", "attacked", "can_move", "can_attack", "blitz", "smokescreen", "fury", "pinned",
                    "attacks", "temp_atk", "temp_hp", "token", "ambush", "shock", "immune", "ambush_ready",
                    "pin_turns", "temp_move_cost", "temp_traits", "temp_removed")
UNIT_VIEW_TYPES = [int, int, int, int, int, bool, int, int, bool, bool, bool, bool, bool, bool, bool, bool, bool,
                   int, int, int, bool, bool, bool, bool, bool, int, int, int, int]
PENDING_VIEW_FIELDS = ("card", "effect", "action", "amount", "select", "side", "kind", "atk", "hp", "previews")
PENDING_VIEW_TYPES = [int, int, str, int, str, str, str, int, int, tuple]
N_CHOOSE = 3 * CONFIG.zone_capacity + 2
INT_FIELDS = ("player", "round", "my_deck", "my_coins", "opp_coins", "my_base_hp", "opp_base_hp", "opp_hand_size",
              "my_deck_size", "opp_deck_size", "front_owner", "result", "phase", "turn", "my_coin_bonus",
              "opp_coin_bonus", "my_burned", "opp_burned")
BOOL_FIELDS = ("is_my_turn", "went_first", "done")
COUNT_FIELDS = ("my_played", "opp_played", "my_decklist", "my_deck_counts", "my_known_hand", "opp_known_hand",
                "opp_revealed", "my_discard", "opp_discard", "my_graveyard", "opp_graveyard")
UNIT_FIELDS = ("my_backline", "opp_backline", "frontline")
ENCODER = ObservationEncoder(CONFIG)
# The Stage 2 encoder (version 4) predates the Stage 3 action space and cannot read its 154-wide mask;
# until the Stage 3 encoder lands, encodings are compared without the mask (the masks themselves are
# compared directly below).
ENCODER_TAKES_MASK = ENCODER_VERSION >= 5


def perturb_hidden(game: Game, observer: int, rng: random.Random, swap_deck: bool = False) -> Game:
    """A clone of `game` that differs only in information hidden from `observer`.

    The opponent's cards the observer knows about (`known_hand[o]`: returned or added cards) stay in its
    hand; the unknown rest of the hand and the deck are redrawn (same sizes, hand kept sorted) from the
    unknown hand cards + deck, or, with `swap_deck`, its deck id (and decklist) is replaced by another deck
    and the unknown cards are drawn from that deck minus the copies already revealed. Both decks are
    reshuffled and the RNG (and seed) reseeded. Public bookkeeping (revealed, known_hand, discard,
    graveyard, history) is left alone.
    """
    g = game.clone()
    o = 1 - observer
    known = Counter({c: k for c, k in enumerate(g.known_hand[o]) if k})
    hand = Counter(g.hands[o])
    assert hand & known == known, "known_hand counts cards that are not in the hand"
    unknown_hand = list((hand - known).elements())
    n_unknown, n_deck = len(unknown_hand), len(g.deck_cards[o])
    if swap_deck:
        new_id = rng.choice([d for d in range(N_DECKS) if d != g.deck_ids[o]])
        ids = list(g.deck_ids)
        ids[o] = new_id
        g.deck_ids = tuple(ids)
        lists = list(g.decklists)
        lists[o] = CONFIG.decks[new_id]
        g.decklists = tuple(lists)
        revealed = Counter({c: k for c, k in enumerate(g.revealed[o]) if k})
        pool = list((Counter(CONFIG.decks[new_id]) - revealed).elements())
    else:
        pool = unknown_hand + g.deck_cards[o]
    assert len(pool) >= n_unknown + n_deck
    rng.shuffle(pool)
    g.hands[o] = sorted(list(known.elements()) + pool[:n_unknown])
    g.deck_cards[o] = pool[n_unknown:n_unknown + n_deck]
    rng.shuffle(g.deck_cards[observer])
    g.rng = random.Random(rng.getrandbits(64))
    if hasattr(g, "seed"):
        g.seed = rng.getrandbits(32)
    g.invalidate()
    return g


def sample_states(seeds, every: int = 1):
    """(seed, game) for every `every`-th state of seeded games over all deck pairs (plus final states)."""
    for seed in seeds:
        decks = DECK_PAIRS[seed % len(DECK_PAIRS)]
        profiles = PROFILE_PAIRS[(7 * seed) % len(PROFILE_PAIRS)]
        for i, (game, _, _) in enumerate(iter_states(seed, profiles, decks)):
            if i % every == 0 or game.done:
                yield seed, game


def encode(game: Game, p: int) -> np.ndarray:
    own = ENCODER_TAKES_MASK and not game.done and game.current_player() == p
    return ENCODER.encode(game.observe(p), game.legal_mask() if own else None)


@pytest.mark.parametrize("swap_deck", [False, True], ids=["same_deck", "other_deck"])
def test_observations_ignore_hidden_information(swap_deck):
    rng = random.Random(2024 + swap_deck)
    checked = opp_view_changed = legal_checked = hand_changed = with_known = 0
    for seed, game in sample_states(range(300, 332), every=3):
        for p in (0, 1):
            alt = perturb_hidden(game, p, rng, swap_deck)
            check_invariants(alt, reachable=False)
            known = Counter({c: k for c, k in enumerate(game.known_hand[1 - p]) if k})
            assert Counter(alt.hands[1 - p]) & known == known  # cards p knows about stay in the hand
            with_known += bool(known)
            obs, alt_obs = game.observe(p), alt.observe(p)
            assert obs == alt_obs, f"seed {seed}: P{p}'s view depends on hidden info\n{game.render()}"
            assert np.array_equal(encode(game, p), encode(alt, p)), f"seed {seed}: encoding leaks"
            if not game.done and game.current_player() == p:
                assert alt.legal_actions() == game.legal_actions()
                assert np.array_equal(alt.legal_mask(), game.legal_mask())
                legal_checked += 1
            # Teeth: the perturbation really changes what the *other* player sees.
            opp_view_changed += alt.observe(1 - p) != game.observe(1 - p)
            hand_changed += alt.hands[1 - p] != game.hands[1 - p]
            checked += 1
    assert checked >= 1000 and legal_checked >= 400, (checked, legal_checked)
    assert opp_view_changed >= (0.99 if swap_deck else 0.4) * checked, (opp_view_changed, checked)
    assert hand_changed >= 0.4 * checked, (hand_changed, checked)
    assert with_known >= 20, with_known  # returned / added cards (Rout, Fall Back, Fire Mission ...) occur


def _reveals(before: Game, after: Game, p: int) -> bool:
    """Whether a step legitimately showed p a card that differs between the hidden-info variants: p drew
    from its own (reshuffled) deck, or a card left the opponent's hand by a random discard."""
    o = 1 - p
    return (len(after.deck_cards[p]) < len(before.deck_cards[p])
            or sum(after.discard[o]) > sum(before.discard[o]))


def test_hidden_information_stays_hidden_through_own_turn():
    """Playing the same actions in two hidden-info variants keeps the actor's view identical, including
    the opponent's draw at the start of their turn and random effects. The RNG is hidden information too
    (the static tests above reseed it), but here both variants share its state, so random effects
    resolve alike and the test isolates the hand and deck. A step that draws a card for the actor (its
    deck order is hidden) or discards a random card from the opponent's hand legitimately shows a card
    that differs between the variants; the comparison stops there for that turn."""
    rng = random.Random(7)
    turns = full = 0
    for seed, game in sample_states(range(400, 416), every=5):
        if game.done:
            continue
        p = game.current_player()
        a, b = game.clone(), perturb_hidden(game, p, rng, swap_deck=turns % 2 == 1)
        b.rng.setstate(a.rng.getstate())
        policy = WeightedPolicy(seed)
        revealed = False
        while not a.done and a.current_player() == p:
            assert a.observe(p) == b.observe(p) and a.legal_actions() == b.legal_actions()
            action = policy(a)
            before = a.clone()
            a.step(action)
            b.step(action)
            if _reveals(before, a, p):
                revealed = True
                break
        turns += 1
        if revealed:
            continue
        assert a.observe(p) == b.observe(p)
        assert np.array_equal(encode(a, p), encode(b, p))
        assert a.done == b.done and a.winner() == b.winner()
        full += 1
    assert turns >= 100 and full >= 0.75 * turns, (turns, full)


def walk(value, path="obs"):
    """Yield (path, leaf) for every leaf of a nested observation."""
    if isinstance(value, tuple):
        for i, item in enumerate(value):
            yield from walk(item, f"{path}[{i}]")
    else:
        yield path, value


def test_observation_fields_and_types_cannot_carry_hidden_cards():
    assert Observation._fields == SPEC_FIELDS
    assert UnitView._fields == UNIT_VIEW_FIELDS
    assert PendingView._fields == PENDING_VIEW_FIELDS
    n = pending = 0
    for _, game in sample_states(range(500, 508), every=3):
        for p in (0, 1):
            o = 1 - p
            obs = game.observe(p)
            assert isinstance(obs, Observation) and isinstance(obs, tuple)
            for name in INT_FIELDS:
                assert type(getattr(obs, name)) is int, (name, type(getattr(obs, name)))
            for name in BOOL_FIELDS:
                assert type(getattr(obs, name)) is bool, name
            assert type(obs.hand) is tuple and all(type(c) is int for c in obs.hand)
            for name in COUNT_FIELDS:
                v = getattr(obs, name)
                assert type(v) is tuple and len(v) == N_CARDS and all(type(x) is int for x in v), name
            assert obs.mulligan_marks == ()  # mulligan off
            # SPEC 5: the pending choice is public (the played card is revealed): the same for both players
            assert obs.pending == game.observe(o).pending
            if obs.pending is not None:
                assert type(obs.pending) is PendingView
                assert [type(x) for x in obs.pending] == PENDING_VIEW_TYPES, obs.pending
                previews = obs.pending.previews
                assert len(previews) == N_CHOOSE and all(
                    type(t) is tuple and len(t) == 4 and all(type(x) is int for x in t) for t in previews)
                pending += 1
            # public history counters: this turn then this game, mirrored in the opponent's view
            for name in ("my_history", "opp_history"):
                v = getattr(obs, name)
                assert type(v) is tuple and len(v) == 6 and all(type(x) is int and x >= 0 for x in v), name
            assert (obs.my_history, obs.opp_history) == (game.observe(o).opp_history, game.observe(o).my_history)
            for name in UNIT_FIELDS:
                zone = getattr(obs, name)
                assert type(zone) is tuple and all(type(u) is UnitView for u in zone), name
                for u in zone:
                    assert [type(x) for x in u] == UNIT_VIEW_TYPES, (name, u)
            # Only plain immutable leaves; nothing that could alias engine objects.
            for path, leaf in walk(obs):
                assert type(leaf) in (int, bool, str, type(None)), (path, type(leaf))
            hash(obs)

            # The only card identities in the view: own hand + units on the (public) board +
            # the public played counts. The opponent's deck id appears nowhere.
            board = all_units(game)
            seen = Counter(obs.hand) + Counter(u.card for u in (*obs.my_backline, *obs.opp_backline,
                                                                 *obs.frontline))
            assert seen == Counter(game.hands[p]) + Counter(u.card for u in board)
            assert obs.opp_hand_size == len(game.hands[o]) <= H
            assert obs.opp_played == tuple(game.played[o]) and obs.my_played == tuple(game.played[p])
            # every opponent non-token unit on the board was played by the opponent (effects only summon tokens)
            on_board = Counter(u.card for u in board if u.owner == o and not u.token)
            assert all(obs.opp_played[c] >= k for c, k in on_board.items())
            n += 1
    assert n >= 200 and pending >= 4, (n, pending)


def expected_pin_turns(game: Game, u) -> int:
    """SPEC 2.4 / 5: the owner turns, from the current turn on, before the END_TURN that lifts the pin (the
    first END_TURN of a turn index >= pin_until), counted turn by turn."""
    if not u.pinned:
        return 0
    last = max(game.turn, u.pin_until)
    return sum(1 for k in range(game.turn, last + 1)
               if (game.current if (k - game.turn) % 2 == 0 else 1 - game.current) == u.owner)


def expected_view(game: Game, u) -> UnitView:
    """SPEC 5 UnitView of a unit, field by field. The armor bit of temp_traits / temp_removed stands for a
    "turn" armor grant (temp_armor > 0) / removal (temp_armor < 0)."""
    return UnitView(u.card, u.atk, u.hp, u.max_hp, u.armor, u.defense, u.nature, u.move_cost, u.summoned, u.moved,
                    u.attacked, u.can_move(), u.can_attack(), u.blitz, u.smokescreen, u.fury, u.pinned, u.attacks,
                    u.temp_atk, u.temp_hp, u.token, u.ambush, u.shock, u.immune, u.ambush_ready,
                    expected_pin_turns(game, u), u.temp_move_cost,
                    u.temp_traits | (ARMOR_BIT if u.temp_armor > 0 else 0),
                    u.temp_removed | (ARMOR_BIT if u.temp_armor < 0 else 0))


def test_observation_is_egocentric_and_correct():
    seen = Counter()
    for _, game in sample_states(range(600, 608), every=3):
        for p in (0, 1):
            o = 1 - p
            obs = game.observe(p)
            fo = game.front_owner
            w = game.winner()
            assert obs.player == p
            assert obs.is_my_turn == (not game.done and game.current_player() == p)
            assert obs.went_first == (game.first_player == p)
            assert obs.round == game.round and obs.my_deck == game.deck_ids[p]
            assert (obs.my_coins, obs.opp_coins) == (game.coins[p], game.coins[o])
            assert (obs.my_base_hp, obs.opp_base_hp) == (game.base_hp[p], game.base_hp[o])
            assert obs.hand == tuple(game.hands[p])
            assert (obs.my_deck_size, obs.opp_deck_size) == (len(game.deck_cards[p]), len(game.deck_cards[o]))
            for zone, units in ((obs.my_backline, game.backline[p]), (obs.opp_backline, game.backline[o]),
                                (obs.frontline, game.frontline)):
                assert zone == tuple(expected_view(game, u) for u in units)
                seen["pinned"] += sum(u.pinned for u in units)
            assert obs.front_owner == (0 if fo is None else (1 if fo == p else -1))
            assert obs.done == game.done
            assert obs.result == (0 if w in (None, DRAW) else (1 if w == p else -1))
            assert (obs.phase, obs.turn) == (game.phase, game.turn)
            assert (obs.my_coin_bonus, obs.opp_coin_bonus) == (game.coin_bonus[p], game.coin_bonus[o])
            assert obs.my_decklist == tuple(Counter(game.decklists[p]).get(c, 0) for c in range(N_CARDS))
            assert obs.my_deck_counts == tuple(Counter(game.deck_cards[p]).get(c, 0) for c in range(N_CARDS))
            assert (obs.my_known_hand, obs.opp_known_hand) == (tuple(game.known_hand[p]), tuple(game.known_hand[o]))
            assert obs.opp_revealed == tuple(game.revealed[o])
            assert (obs.my_discard, obs.opp_discard) == (tuple(game.discard[p]), tuple(game.discard[o]))
            assert (obs.my_graveyard, obs.opp_graveyard) == (tuple(game.graveyard[p]), tuple(game.graveyard[o]))
            assert (obs.my_burned, obs.opp_burned) == (game.burned[p], game.burned[o])
            assert obs.my_history == (*game.history_turn[p], *game.history_game[p])
            assert obs.opp_history == (*game.history_turn[o], *game.history_game[o])
            assert obs.pending == game.observe(o).pending and (obs.pending is None) == (game.pending is None)
            for q, played, hist in ((p, obs.my_played, obs.my_history), (o, obs.opp_played, obs.opp_history)):
                board = Counter(u.card for u in all_units(game) if u.owner == q and not u.token)
                ops = sum(k for c, k in enumerate(played) if CONFIG.cards[c].is_operation)
                assert hist[3:5] == (ops, sum(played) - ops) and hist[5] == sum(game.graveyard[q])
                for c in range(N_CARDS):
                    card = CONFIG.cards[c]
                    if card.token:  # tokens are only created by effects and never revealed
                        assert game.revealed[q][c] == 0
                    elif card.is_operation:  # played operations go to the discard pile, never to the board
                        assert board[c] == 0 and game.discard[q][c] >= played[c]
                    else:  # a non-token unit is on the board or dead only after being played from the hand
                        assert board[c] + game.graveyard[q][c] <= played[c]
                    # a card is revealed when it is played or discarded unknown
                    assert game.revealed[q][c] <= played[c] + game.discard[q][c]
    assert seen["pinned"] > 0, seen  # "turn" trait changes: tests/test_features.py (the shipped pool has none)


def test_observation_is_immutable_and_does_not_alias_engine_state():
    game = None
    for g, _, _ in iter_states(31, ("full_frontline", "defense_wall"), (1, 2)):
        if g.backline[0] and g.backline[1] and g.frontline and not g.done:
            game = g.clone()
            break
    assert game is not None
    p = game.current_player()
    before = snapshot(game)
    obs = game.observe(p)
    other = game.observe(1 - p)
    assert snapshot(game) == before  # observing is side-effect free
    frozen, frozen_other = copy.deepcopy(obs), copy.deepcopy(other)

    with pytest.raises(AttributeError):
        obs.round = 99
    with pytest.raises(TypeError):
        obs.hand[0] = 0
    with pytest.raises(TypeError):
        obs.my_played[0] = 5
    with pytest.raises(AttributeError):
        obs.frontline[0].hp = 99

    # Mutate the game in place, then keep playing: the old observations must not move.
    for u in all_units(game):
        u.hp += 7
        u.summoned = not u.summoned
        u.armor += 1
    if game.hands[p]:
        game.hands[p].pop()
    game.deck_cards[p].clear()
    game.played[p][0] += 1
    game.coins[p] += 5
    game.base_hp[1 - p] -= 1
    game.invalidate()
    for _ in range(30):
        if game.done:
            break
        game.step(game.legal_actions()[-1])
    assert obs == frozen and other == frozen_other


def test_encoder_uses_only_the_observation_and_mask():
    # The encoding is a pure function of (Observation, mask): equal inputs, equal vectors,
    # regardless of which Game produced them.
    g1, g2 = new_game(77), new_game(77)
    rng = random.Random(1)
    p = g1.current_player()
    alt = perturb_hidden(g2, p, rng, swap_deck=True)
    assert alt.deck_ids[1 - p] != g1.deck_ids[1 - p]
    assert np.array_equal(encode(g1, p), encode(alt, p))
    if ENCODER_TAKES_MASK:
        assert np.array_equal(ENCODER.encode(g1.observe(p), g1.legal_mask()),
                              ENCODER.encode(alt.observe(p), alt.legal_mask()))
    batch = ENCODER.encode_batch([g1.observe(0), g1.observe(1)])
    assert np.array_equal(batch[0], ENCODER.encode(g1.observe(0)))
    assert np.array_equal(batch[1], ENCODER.encode(g1.observe(1)))
