from tuner.bench import compare_quality, percentile, quality_answer_correct


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 3.0], 0.5) == 2.0


def test_quality_allows_nonexact_correct_text() -> None:
    reference = {"rows": [{"prompt": "p", "ok": True, "text": "same", "correct": True}]}
    candidate = {
        "all_ok": True,
        "rows": [{"prompt": "p", "ok": True, "text": "different", "correct": True}],
    }
    assert compare_quality(candidate, reference)["quality_safe"]
    candidate["rows"][0]["correct"] = False
    assert not compare_quality(candidate, reference)["quality_safe"]


def test_semantic_quality_examples() -> None:
    assert quality_answer_correct(0, "1591")
    assert quality_answer_correct(2, "No glib can be a flan because every glib is a trob.")
    assert quality_answer_correct(4, "TCP is reliable and ordered; UDP reduces latency.")
    assert quality_answer_correct(11, "2^10 is larger: 1024 versus 1000.")
