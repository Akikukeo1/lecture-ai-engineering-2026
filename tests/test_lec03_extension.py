"""第3回追加ツールと評価シナリオのAPI不要テスト。"""

import json

from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent


def test_quiz_uses_a_real_entry_and_is_deterministic():
    result = json.loads(tools.make_quiz(2))
    assert result == json.loads(tools.make_quiz(2))
    entries = [item for item in tools.load_glossary() if item["lecture"] == 2]
    assert any(
        result == {
            "question": f"「{item['term']}」の意味は？",
            "answer": item["definition"],
        }
        for item in entries
    )


def test_quiz_handles_unknown_lecture():
    assert "見つかりません" in tools.make_quiz(0)
    assert "見つかりません" in tools.make_quiz(99)


def test_quiz_is_registered_and_invocable_without_api_key():
    model = TestModel(call_tools=["make_quiz"])
    with agent.override(model=model):
        result = agent.run_sync("第2回の確認問題を出して")
    names = {
        tool.name
        for tool in model.last_model_request_parameters.function_tools
    }
    assert "make_quiz" in names
    assert result.output


def test_quiz_documented_for_model():
    assert "確認問題" in (tools.make_quiz.__doc__ or "")
