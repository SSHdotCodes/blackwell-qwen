from tuner.bench import compare_quality, percentile


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 3.0], 0.5) == 2.0


def test_quality_requires_exact_text() -> None:
    reference = {"rows": [{"prompt": "p", "ok": True, "text": "same"}]}
    candidate = {"rows": [{"prompt": "p", "ok": True, "text": "same"}]}
    assert compare_quality(candidate, reference)["quality_safe"]
    candidate["rows"][0]["text"] = "different"
    assert not compare_quality(candidate, reference)["quality_safe"]

