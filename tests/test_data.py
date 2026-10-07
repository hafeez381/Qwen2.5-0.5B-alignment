from alignment.data import hh_row, row_sha, uf_row, ultrachat_row


def test_hh_single_turn_is_parsed():
    ex = {"chosen": "\n\nHuman: How do I boil an egg?\n\nAssistant: Simmer it for 9 minutes.",
          "rejected": "\n\nHuman: How do I boil an egg?\n\nAssistant: No idea."}
    assert hh_row(ex) == {"prompt": "How do I boil an egg?", "chosen": "Simmer it for 9 minutes.", "rejected": "No idea."}


def test_hh_multi_turn_mismatched_and_identical_rows_are_rejected():
    two_turns = "\n\nHuman: Hi\n\nAssistant: Hello\n\nHuman: Egg?\n\nAssistant: "
    assert hh_row({"chosen": two_turns + "Boil it.", "rejected": two_turns + "No."}) is None
    assert hh_row({"chosen": "\n\nHuman: A\n\nAssistant: x", "rejected": "\n\nHuman: B\n\nAssistant: y"}) is None
    assert hh_row({"chosen": "\n\nHuman: A\n\nAssistant: x", "rejected": "\n\nHuman: A\n\nAssistant: x"}) is None


def test_uf_drops_ties_and_takes_last_assistant_turn():
    msgs = lambda a: [{"role": "user", "content": "Q"}, {"role": "assistant", "content": a}]
    ex = {"prompt": "Q", "chosen": msgs("good"), "rejected": msgs("bad"), "score_chosen": 8.0, "score_rejected": 3.0}
    assert uf_row(ex) == {"prompt": "Q", "chosen": "good", "rejected": "bad"}
    assert uf_row({**ex, "score_rejected": 8.0}) is None


def test_ultrachat_takes_first_turn_pair_and_hash_is_stable():
    ex = {"messages": [{"role": "user", "content": " Q1 "}, {"role": "assistant", "content": "A1"},
                       {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]}
    row = ultrachat_row(ex)
    assert row == {"prompt": "Q1", "response": "A1"}
    assert row_sha(row) == row_sha({"response": "A1", "prompt": "Q1", "id": "ignored"})
