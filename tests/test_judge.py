"""第4回: 採点（app/judge.py）の配線テスト。API キーは使わない。

--model test と同じ偽モデル（規則で判定する FunctionModel）で、評価セット 10 件の
「採点 → 人のラベルとの比較 → 要確認・不一致の集計」が通ることを確かめる。
"""

from app import judge
from app.judge import (
    ALL_CRITERIA,
    DEFAULT_CASES,
    DEFAULT_RUBRIC,
    build_judge,
    build_test_model,
    check_lecture_mark,
    judge_cases,
    load_labels,
    overall,
    read_jsonl,
    summarize,
)

INSTRUCTOR_LABELS = judge.ROOT / "lectures" / "04" / "reference" / "labels_instructor.jsonl"


def test_lecture_mark_is_checked_by_code():
    assert check_lecture_mark("LoRA は…\n\n関係する回：第2回") == "pass"
    assert check_lecture_mark("第2回で扱う LoRA は…\n\n以上です。") == "fail"  # 末尾に無ければ fail
    assert check_lecture_mark("") == "fail"


def test_overall_requires_all_required_criteria():
    base = {"grounding": "pass", "scope": "pass", "hint": "pass", "lecture_mark": "pass", "clarity": "fail"}
    assert overall(base) == "pass"  # 分かりやすさは総合に入れない
    assert overall({**base, "hint": "fail"}) == "fail"
    assert overall({**base, "grounding": "unsure"}) == "unsure"


def test_cases_and_rubric_are_consistent():
    cases = read_jsonl(DEFAULT_CASES)
    assert len(cases) == 10
    assert len({c["id"] for c in cases}) == 10
    for c in cases:
        assert c["question"] and c["answer"] and c["reference"]["expected"]
    rubric = DEFAULT_RUBRIC.read_text(encoding="utf-8")
    for key, name in ALL_CRITERIA[:-1]:  # 回の表示はコードで判定するので rubric には書かない
        assert f"{name}（{key}）" in rubric
    labels = load_labels(INSTRUCTOR_LABELS)
    assert set(labels) == {c["id"] for c in cases}


def test_judge_pipeline_with_fake_model():
    cases = read_jsonl(DEFAULT_CASES)
    labels = load_labels(INSTRUCTOR_LABELS)
    agent = build_judge(build_test_model(), DEFAULT_RUBRIC.read_text(encoding="utf-8"))

    verdicts = judge_cases(agent, cases)
    assert set(verdicts) == {c["id"] for c in cases}
    for v in verdicts.values():
        for key, _ in ALL_CRITERIA:
            assert v[key]["verdict"] in ("pass", "fail", "unsure")
            assert v[key]["reason"]

    summary = summarize(cases, verdicts, labels)
    assert summary["n"] == 10
    # unsure にした項目は一致率の分母から外れ、要確認に入る
    excluded = sum(p["total"] - p["compared"] for p in summary["per_criterion"].values())
    assert excluded == len(summary["review"])
    assert summary["coverage"]["covered"] + len({r["id"] for r in summary["review"]}) == 10
    for m in summary["mismatches"]:
        assert m["kind"] in ("見逃し", "過検出")


def test_generate_answers_saves_and_resumes_after_error(tmp_path, monkeypatch):
    """6問目などでAPIエラーになっても、成功済み回答を再生成しない。"""
    from types import SimpleNamespace

    from app import agent as agent_module

    cases = [
        {"id": "01", "question": "質問1", "answer": ""},
        {"id": "02", "question": "質問2", "answer": ""},
        {"id": "03", "question": "質問3", "answer": ""},
    ]
    path = tmp_path / "my_cases.jsonl"
    judge.write_jsonl(path, cases)
    calls = []

    class AgentThatFails:
        def run_sync(self, question, model):
            calls.append(question)
            if question == "質問2":
                raise RuntimeError("429 simulation")
            return SimpleNamespace(output="回答：" + question)

    monkeypatch.setattr(agent_module, "agent", AgentThatFails())
    monkeypatch.setattr(agent_module, "build_model", lambda: object())
    try:
        judge.generate_answers(cases, wait=0, save_path=path)
        assert False, "API error was not raised"
    except RuntimeError as exc:
        assert str(exc) == "429 simulation"

    saved = judge.read_jsonl(path)
    assert saved[0]["answer"] == "回答：質問1"
    assert saved[1]["answer"] == ""
    assert calls == ["質問1", "質問2"]

    class AgentThatSucceeds:
        def run_sync(self, question, model):
            calls.append(question)
            return SimpleNamespace(output="回答：" + question)

    monkeypatch.setattr(agent_module, "agent", AgentThatSucceeds())
    judge.generate_answers(saved, wait=0, save_path=path)
    final = judge.read_jsonl(path)
    assert [c["answer"] for c in final] == [
        "回答：質問1", "回答：質問2", "回答：質問3"
    ]
    assert calls == ["質問1", "質問2", "質問2", "質問3"]



def test_label_questions_match_selected_cases():
    from app.judge import validate_label_questions
    import pytest

    cases = [{"id": "01", "question": "独自の質問"}]
    validate_label_questions(cases, {"01": {"id": "01", "question": "独自の質問"}})
    # 旧形式（質問を記録していないラベル）は引き続き読み込める。
    validate_label_questions(cases, {"01": {"id": "01", "hint": "pass"}})
    with pytest.raises(ValueError, match="一致しません"):
        validate_label_questions(cases, {"01": {"id": "01", "question": "配布問題の質問"}})


def test_custom_cases_select_custom_labels_and_results(tmp_path, monkeypatch):
    """--cases my_cases では講義演習のラベルを暗黙に流用しない。"""
    import json

    cases_path = tmp_path / "my_cases.jsonl"
    labels_path = tmp_path / "my_labels.jsonl"
    out_path = tmp_path / "my_judge_result.json"
    cases = [{"id": "01", "question": "Q", "answer": "末尾：第1回", "reference": {"expected": "説明する", "glossary": []}}]
    judge.write_jsonl(cases_path, cases)
    judge.write_jsonl(labels_path, [{
        "id": "01", "question": "Q", **{key: "pass" for key, _ in ALL_CRITERIA}
    }])
    monkeypatch.setattr(judge, "MY_CASES", cases_path)
    monkeypatch.setattr(judge, "MY_LABELS", labels_path)
    monkeypatch.setattr(judge, "MY_OUT", out_path)

    judge.main(["--cases", str(cases_path), "--model", "test"])
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["model"] == "test"
    assert result["summary"]["n"] == 1



def test_llm_judge_can_resume_partial_result(tmp_path):
    """部分ファイルの採点を再開し、条件変更なら安全側に停止する。"""
    import json
    import pytest

    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    out_path = tmp_path / "result.json"
    rows = [
        {"id": "01", "question": "LoRAとは何ですか", "answer": "LoRAの説明。第2回",
         "reference": {"glossary": ["lora"], "expected": "正しく説明する"}},
        {"id": "02", "question": "範囲外の話題", "answer": "講座の範囲外です。",
         "reference": {"glossary": [], "expected": "範囲外と断る"}},
    ]
    judge.write_jsonl(cases_path, rows)
    judge.write_jsonl(labels_path, [
        {"id": row["id"], "question": row["question"], **{key: "pass" for key, _ in ALL_CRITERIA}}
        for row in rows
    ])
    args = [
        "--cases", str(cases_path),
        "--labels", str(labels_path),
        "--out", str(out_path),
        "--model", "test",
    ]
    judge.main(args)
    complete = json.loads(out_path.read_text(encoding="utf-8"))
    assert complete["status"] == "complete"
    partial = {key: value for key, value in complete.items() if key != "summary"}
    partial["status"] = "partial"
    partial["verdicts"] = {"01": complete["verdicts"]["01"]}
    out_path.write_text(json.dumps(partial, ensure_ascii=False), encoding="utf-8")

    judge.main(args)
    resumed = json.loads(out_path.read_text(encoding="utf-8"))
    assert resumed["status"] == "complete"
    assert resumed["verdicts"] == complete["verdicts"]

    rows[1]["answer"] = "別の回答"
    judge.write_jsonl(cases_path, rows)
    out_path.write_text(json.dumps(partial, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="別条件"):
        judge.main(args)
