"""第5回のハイブリッド検索: APIキーを使わない単体テスト。"""

import json

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
