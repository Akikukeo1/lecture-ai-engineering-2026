"""第4回レポート集計のユニットテスト（APIキー不要）。"""
import pytest

from scripts.lec04_report import analyze, fraction, markdown


def sample():
    cases = [{"id": "01"}, {"id": "02"}, {"id": "03"}]
    labels = []
    verdicts = {}
    for i in range(3):
        cid = f"{i+1:02d}"
        labels.append({"id": cid, **{key: ("fail" if i < 2 else "pass") for key in (
            "grounding", "scope", "hint", "clarity", "lecture_mark"
        )}})
        verdicts[cid] = {key: {"verdict": ["fail", "pass", "unsure"][i], "reason": "確認した"} for key in (
            "grounding", "scope", "hint", "clarity", "lecture_mark"
        )}
    return cases, labels, verdicts


def test_missed_fail_and_unsure_do_not_inflate_agreement():
    result = analyze(*sample())
    m = result["metrics"]["grounding"]
    assert (m["total"], m["compared"], m["agree"], m["unsure"]) == (3, 2, 1, 1)
    assert (m["human_fail"], m["caught_fail"], m["missed_fail"], m["false_alarm"]) == (2, 1, 1, 0)
    assert len(result["disagreements"]) == 5
    assert "1/2 (50.0%)" in markdown(result, "gemini-example")
    assert "算出不可" not in fraction(0, 1)


def test_zero_denominator_is_not_reported_as_full_recall():
    assert fraction(0, 0) == "算出不可（分母0）"


def test_missing_labels_are_rejected():
    cases, labels, verdicts = sample()
    with pytest.raises(ValueError, match="ID集合"):
        analyze(cases, labels[:2], verdicts)


def test_unsettled_label_is_rejected():
    cases, labels, verdicts = sample()
    labels[0]["hint"] = None
    with pytest.raises(ValueError, match="未確定"):
        analyze(cases, labels, verdicts)


def test_duplicate_case_id_is_rejected():
    cases, labels, verdicts = sample()
    cases[1]["id"] = "01"
    with pytest.raises(ValueError, match="重複"):
        analyze(cases, labels, verdicts)


def test_report_rejects_labels_for_another_question():
    cases, labels, verdicts = sample()
    labels[0]["question"] = "別の質問"
    cases[0]["question"] = "本来の質問"
    with pytest.raises(ValueError, match="質問が一致"):
        analyze(cases, labels, verdicts)
