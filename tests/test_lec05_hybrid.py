"""第5回のハイブリッド検索: APIキーを使わない単体テスト。"""

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app import hybrid


@pytest.fixture
def sample_pages(monkeypatch, tmp_path):
    def page(page_id, document_id, section, title, content):
        return {
            "page_id": page_id, "document_id": document_id, "lecture": 2,
            "document_title": document_id, "section": section,
            "title": title, "page": int(page_id[-2:]), "content": content,
        }

    pages = [
        page("d1-p01", "d1", "LoRA", "LoRAの仕組み", "元の重みを固定して差分を学習する。"),
        page("d1-p02", "d1", "LoRA", "QLoRAと量子化", "QLoRAは元の重みを4bit量子化する。"),
        page("d1-p03", "d1", "別の章", "エージェント", "APIでツールを呼び出す。"),
        page("d2-p01", "d2", "別資料", "ツールの使い方", "APIを呼び出す。"),
        page("d2-p02", "d2", "別資料", "プロンプト", "指示文を工夫する。"),
    ]
    graph = {
        "nodes": [
            *({"id": p["page_id"], "kind": "page", "title": p["title"]} for p in pages),
            {"id": "term-qlora", "kind": "term", "title": "QLoRA"},
            {"id": "outside-p99", "kind": "page", "title": "対象外ページ"},
        ],
        "edges": [
            {"source": "d1-p01", "target": "d1-p02", "relation": "next_page"},
            {"source": "d1-p01", "target": "term-qlora", "relation": "mentions"},
            {"source": "d1-p02", "target": "term-qlora", "relation": "mentions"},
            {"source": "outside-p99", "target": "term-qlora", "relation": "mentions"},
        ],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(hybrid, "RAG_GRAPH", graph_path)
    monkeypatch.setattr(hybrid, "load_pages", lambda: pages)
    monkeypatch.setattr(
        hybrid, "search", lambda _query: [
            {"page_id": "d1-p01", "distance": 0.10},
            {"page_id": "d1-p03", "distance": 0.20},
            {"page_id": "d2-p01", "distance": 0.30},
            {"page_id": "d2-p02", "distance": 0.40},
        ],
    )
    return pages


def test_related_page_is_selected_from_tree_and_graph(sample_pages):
    result = hybrid.hybrid_search("LoRAとQLoRAの量子化", max_results=4)
    ids = [page["page_id"] for page in result["results"]]
    assert ids[:3] == ["d1-p01", "d1-p03", "d2-p01"]
    assert ids[3] == "d1-p02"
    paths = result["results"][3]["retrieval_paths"]
    assert "tree:section" in paths
    assert "graph:adjacent" in paths
    assert "graph:term" in paths


def test_budget_deduplication_and_no_out_of_scope_pages(sample_pages):
    result = hybrid.hybrid_search("LoRAとQLoRAの量子化", max_results=4)
    ids = [page["page_id"] for page in result["results"]]
    assert len(ids) == len(set(ids)) <= 4
    assert "outside-p99" not in ids


def test_no_added_page_without_keyword_evidence(sample_pages):
    result = hybrid.hybrid_search("未知の専門用語abcdefgh", max_results=5)
    assert len(result["results"]) == 3


def test_feature_extraction_preserves_japanese_and_english():
    terms = hybrid.features("QLoRAは4bit量子化、メモリ")
    assert "qlora" in terms
    assert "4bit" in terms
    assert "量子" in terms
    assert "メモリ" in terms


def test_rejects_empty_query_and_invalid_budget(sample_pages):
    with pytest.raises(ValueError):
        hybrid.hybrid_search("  ")
    with pytest.raises(ValueError):
        hybrid.hybrid_search("LoRA", max_results=0)


@pytest.mark.parametrize(
    ("cost", "expected"),
    [(Decimal("0.0013181"), "0.0013181"), (None, None)],
)
def test_record_serializes_generation_cost_as_json(cost, expected):
    """比較ログはGeminiのDecimal型費用を精度を落とさず記録できる。"""
    result = SimpleNamespace(
        all_messages=lambda: [],
        output="資料に基づく回答",
        usage=SimpleNamespace(
            input_tokens=3027,
            output_tokens=164,
            requests=1,
            cost=cost,
        ),
    )
    record = hybrid._record(result, "LoRAとは？", "vector", 2.913, [])
    decoded = json.loads(json.dumps([record], ensure_ascii=False))
    assert decoded[0]["estimated_generation_usd"] == expected
    assert decoded[0]["input_tokens"] == 3027



def test_compare_resume_only_retries_failed_case(monkeypatch, tmp_path):
    """503で失敗した条件のみ再実行し、既存の成功済みログを保持する。"""
    from app import agent as agent_module
    from types import SimpleNamespace

    sample_case = hybrid.CASES[0]
    output = tmp_path / "comparison.json"
    saved_vector = {
        "question": sample_case["question"], "mode": "vector", "model": hybrid.MODEL,
        "answer": "すでに得られた実測", "elapsed_seconds": 3.995,
    }
    failed_hybrid = {
        "question": sample_case["question"], "mode": "hybrid", "error": "HTTP 503",
    }
    output.write_text(json.dumps([saved_vector, failed_hybrid], ensure_ascii=False),
                      encoding="utf-8")

    called = []

    class FakeAgent:
        def __init__(self, mode):
            self.mode = mode

        def run_sync(self, question, *, model):
            called.append((self.mode, question))
            return SimpleNamespace(
                all_messages=lambda: [],
                output="改良版の再実行結果",
                usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                                      requests=1, cost=Decimal("0.0001234")),
            )

    monkeypatch.setattr(agent_module, "agent", FakeAgent("vector"))
    monkeypatch.setattr(hybrid, "HYBRID_AGENT", FakeAgent("hybrid"))
    monkeypatch.setattr(hybrid, "build_model", lambda: object())

    hybrid.compare(output, 1, resume=True)
    actual = json.loads(output.read_text(encoding="utf-8"))

    assert called == [("hybrid", sample_case["question"])]
    assert len(actual) == 2
    assert actual[0] == saved_vector
    assert actual[1]["mode"] == "hybrid"
    assert actual[1]["estimated_generation_usd"] == "0.0001234"
    assert "error" not in actual[1]


def test_resume_rejects_mismatched_model(monkeypatch, tmp_path):
    """モデルを変更した場合は既存の結果と混在させない。"""
    case = hybrid.CASES[0]
    output = tmp_path / "comparison.json"
    output.write_text(json.dumps([
        {"question": case["question"], "mode": "vector",
         "model": "a-different-model", "answer": "old"},
    ]), encoding="utf-8")

    with pytest.raises(ValueError, match="モデル"):
        hybrid.compare(output, 1, resume=True)
