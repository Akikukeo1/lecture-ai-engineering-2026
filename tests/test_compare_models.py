"""Gemini APIを呼ばずに、第1回発展課題の計測・出力を検証する。"""

import csv
from pathlib import Path
from types import SimpleNamespace

from app.compare_models import measure, save_csv


class FakeAPIError(Exception):
    def __init__(self, code: int):
        self.code = code
        super().__init__(str(code))


class FakeClient:
    def __init__(self, fail_at: int | None = None):
        self.calls: list[tuple[str, str]] = []
        self.fail_at = fail_at
        self.models = self

    def generate_content(self, *, model: str, contents: str):
        self.calls.append((model, contents))
        if self.fail_at == len(self.calls):
            raise FakeAPIError(429)
        return SimpleNamespace(
            text=f"回答: {model} / {contents}",
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=20,
                total_token_count=35,
            ),
        )


def test_model_question_combinations_and_csv(tmp_path: Path):
    client = FakeClient()
    readings = iter([0.0, 1.25, 2.0, 2.75, 3.0, 4.5, 5.0, 6.0])
    slept: list[float] = []
    rows = measure(
        client,
        ["m1", "m2"],
        ["q1", "q2"],
        delay=2.0,
        clock=lambda: next(readings),
        sleep=slept.append,
        api_error_type=FakeAPIError,
    )
    assert len(rows) == 4
    assert client.calls == [("m1", "q1"), ("m1", "q2"), ("m2", "q1"), ("m2", "q2")]
    assert slept == [2.0, 2.0, 2.0]
    assert rows[0]["elapsed_seconds"] == 1.25
    assert rows[0]["prompt_tokens"] == 10
    assert rows[0]["output_tokens"] == 20
    assert rows[0]["total_tokens"] == 35
    assert "回答:" in rows[0]["response_text"]

    output = tmp_path / "results" / "compare.csv"
    save_csv(rows, output)
    with output.open(encoding="utf-8-sig", newline="") as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == 4
    assert saved[0]["elapsed_seconds"] == "1.25"
    assert saved[0]["status"] == "ok"


def test_rate_limit_stops_remaining_requests():
    client = FakeClient(fail_at=2)
    readings = iter([0, 1, 2, 3])
    rows = measure(
        client,
        ["m1", "m2"],
        ["q1", "q2"],
        delay=0,
        clock=lambda: next(readings),
        api_error_type=FakeAPIError,
    )
    assert len(client.calls) == 2
    assert len(rows) == 2
    assert rows[-1]["status"] == "http_429"
    assert rows[-1]["response_text"] == ""
    assert rows[-1]["prompt_tokens"] is None


def test_missing_usage_metadata():
    client = FakeClient()
    def without_metadata(*, model, contents):
        return SimpleNamespace(text="回答", usage_metadata=None)
    client.models.generate_content = without_metadata
    rows = measure(client, ["m1"], ["q1"], delay=0, api_error_type=FakeAPIError)
    assert rows[0]["status"] == "ok"
    assert rows[0]["prompt_tokens"] is None
