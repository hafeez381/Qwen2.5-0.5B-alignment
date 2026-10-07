from alignment.persona import conversational, split


def test_conversational_records_and_disjoint_seeded_split():
    rows = [{"prompt": f"p{i}", "chosen": f"c{i}", "rejected": f"r{i}"} for i in range(30)]
    tr, ev = split(rows, 5, 42)
    assert len(ev) == 5 and len(tr) == 25 and not set(tr) & set(ev)
    assert split(rows, 5, 42) == (tr, ev)
    ds = conversational(rows[:2])
    assert ds[0]["prompt"] == [{"role": "user", "content": "p0"}]
    assert ds[0]["chosen"] == [{"role": "assistant", "content": "c0"}]
    assert ds[0]["rejected"] == [{"role": "assistant", "content": "r0"}]
