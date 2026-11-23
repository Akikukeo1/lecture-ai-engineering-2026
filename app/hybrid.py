"""第5回発展: 章の構造とGraphを併用した根拠重視の検索。

ベクトル検索の上位ページを起点に、同じ章・前後ページ・共通用語の
候補を集め、質問に含まれる特徴語の重み付き一致度で予算内に絞る。
埋め込みと教材を変更せず、既存の通常RAGとの比較を可能にする。
"""

import argparse
from collections import Counter, defaultdict
import json
import math
import re
from pathlib import Path
from time import perf_counter

from google.genai.errors import APIError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from app.agent import INSTRUCTIONS as BASE_INSTRUCTIONS, build_model
from app.config import MODEL, RAG_GRAPH, api_error_message
from app.retrieval import load_pages, search

MAX_RESULTS = 5
SEED_COUNT = 3

# 日本語は外部分かち書き器に依存せず2〜4文字の連続文字列で照合する。
# 英数字の専門用語は単語単位で保持する。
WORDS = re.compile(r"[一-龥々ぁ-んァ-ヴー]+|[a-z0-9]+(?:[-_.][a-z0-9]+)*", re.IGNORECASE)


def features(text: str) -> set[str]:
    """英数字の用語と日本語文字N-gram（2〜4文字）を返す。"""
    result = set()
    for word in WORDS.findall(text.casefold()):
        if word.isascii():
            if len(word) >= 2:
                result.add(word)
        else:
            for size in (2, 3, 4):
                result.update(word[i:i + size] for i in range(len(word) - size + 1))
    return result


def relevance(query_features: set[str], title_features: set[str],
              body_features: set[str], document_frequency: Counter, total_pages: int) -> float:
    """資料全体で珍しい質問語ほど重みを付け、タイトルの一致を優先する。"""
    if not query_features:
        return 0.0

    def weight(feature: str) -> float:
        return math.log1p((total_pages + 1) / (document_frequency[feature] + 1))

    normalizer = sum(weight(term) for term in query_features)
    matched = sum(
        weight(term) * (1.5 if term in title_features else 1.0)
        for term in query_features if term in body_features or term in title_features
    )
    return matched / normalizer if normalizer else 0.0


def hybrid_search(query: str, *, max_results: int = MAX_RESULTS) -> dict:
    """ベクトル検索の入口を保ち、Tree/Graph由来の候補から根拠ページを選ぶ。

    候補に使うのは指定対象資料の本文だけ。Graphの範囲外ノードは採用しない。
    同じページを重複せず、追加候補が質問と一致しなければ件数を増やさない。
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    if max_results < 1:
        raise ValueError("max_resultsは1以上にしてください。")

    pages = {page["page_id"]: page for page in load_pages()}
    hits = [hit for hit in search(query) if hit["page_id"] in pages]
    # 既存の検索DBと資料の対象範囲を共通にし、入口の順位を保持する。
    seeds = hits[:min(SEED_COUNT, max_results)]
    seed_ids = {hit["page_id"] for hit in seeds}
    candidates: dict[str, set[str]] = defaultdict(set)

    def add(page_id: str, source: str) -> None:
        if page_id in pages and page_id not in seed_ids:
            candidates[page_id].add(source)

    # 通常検索で4〜5位だったページも、構造的候補と同じ土俵で比較する。
    for hit in hits[len(seeds):]:
        add(hit["page_id"], "vector")

    # Tree: 同一資料・同一章に属するページを候補にする。
    sections: dict[tuple[str, str], list[str]] = defaultdict(list)
    for page in pages.values():
        sections[(page["document_id"], page["section"])].append(page["page_id"])
    for seed in seeds:
        page = pages[seed["page_id"]]
        for sibling_id in sections[(page["document_id"], page["section"])]:
            add(sibling_id, "tree:section")

    # Graph: 隣のページ(1 hop)、同じ用語に言及するページ(2 hop)。
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    kinds = {node["id"]: node["kind"] for node in graph["nodes"]}
    next_pages: dict[str, set[str]] = defaultdict(set)
    page_terms: dict[str, set[str]] = defaultdict(set)
    term_pages: dict[str, set[str]] = defaultdict(set)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["relation"] == "next_page" and a in pages and b in pages:
            next_pages[a].add(b)
            next_pages[b].add(a)
        elif edge["relation"] == "mentions":
            if a in pages and kinds.get(b) == "term":
                page_terms[a].add(b)
                term_pages[b].add(a)
            elif b in pages and kinds.get(a) == "term":
                page_terms[b].add(a)
                term_pages[a].add(b)

    for seed in seeds:
        root = seed["page_id"]
        for neighbor in next_pages[root]:
            add(neighbor, "graph:adjacent")
        for term in page_terms[root]:
            # 一般用語による候補の爆発は避け、重み付けは後段で行う。
            for related in term_pages[term]:
                add(related, "graph:term")

    # 候補は質問との一致度を計算して並べ、データを取得するだけで満杯にしない。
    q_features = features(query)
    page_features = {
        page_id: (
            features(page["title"] + " " + page["section"]),
            features(page["content"]),
        )
        for page_id, page in pages.items()
    }
    df: Counter = Counter()
    for title, body in page_features.values():
        df.update(title | body)

    ranked = []
    for page_id, paths in candidates.items():
        title, body = page_features[page_id]
        score = relevance(q_features, title, body, df, len(pages))
        if score > 0:
            ranked.append((score, page_id, paths))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    results = [
        {**pages[hit["page_id"]], "distance": hit["distance"],
         "retrieval_paths": ["vector:seed"]}
        for hit in seeds
    ]
    for score, page_id, paths in ranked[:max_results - len(results)]:
        results.append({**pages[page_id], "lexical_score": round(score, 4),
                        "retrieval_paths": sorted(paths)})

    return {
        "results": results,
        "meta": {"seed_ids": [hit["page_id"] for hit in seeds],
                 "candidate_count": len(candidates), "max_results": max_results,
                 "selection": "IDF-weighted Japanese n-grams + English terms"},
    }


def search_evidence_materials(query: str) -> str:
    """第1〜3回の講義資料を、ベクトル・章・関連用語から最大5ページ検索する。

    質問に必要な内容をqueryに含める。retrieval_pathsは候補の取得経路であり、
    正答の保証ではない。本文を確認し、ページ番号を根拠として回答すること。
    """
    try:
        return json.dumps(hybrid_search(query), ensure_ascii=False)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        return json.dumps({"error": str(error), "results": []}, ensure_ascii=False)
    except APIError as error:
        return json.dumps({"error": api_error_message(error.code), "results": []},
                          ensure_ascii=False)


# 比較時の回答方針を揃えるため、通常RAGとの違いはツール名だけにする。
INSTRUCTIONS = BASE_INSTRUCTIONS.replace("search_materials", "search_evidence_materials")

HYBRID_AGENT = Agent(instructions=INSTRUCTIONS)
HYBRID_AGENT.tool_plain(search_evidence_materials)

CASES = [
    {
        "name": "単一ページ",
        "question": "第2回の説明では、LoRAは何を固定し何を学習しますか？ QLoRAは何を加えますか？",
        "reference_pages": ["lec02-main-p052"],
    },
    {
        "name": "複数ページ",
        "question": "第3回の例では、コンパクションとセッションをまたぐメモリはどう違いますか。日々のメモはすべて長期メモリに入りますか？",
        "reference_pages": ["lec03-main-p013", "lec03-main-p039", "lec03-main-p040"],
    },
    {
        "name": "資料にない",
        "question": "今回作ったRAGの実測平均応答時間と正答率を教えてください。",
        "reference_pages": [],
    },
]


def _record(result, question: str, mode: str, elapsed: float, reference: list[str]) -> dict:
    calls = []
    retrieved = []
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                calls.append({"name": part.tool_name, "args": str(part.args)})
            elif isinstance(part, ToolReturnPart):
                try:
                    data = json.loads(part.content)
                except (TypeError, ValueError):
                    continue
                if isinstance(data, dict):
                    retrieved.extend(page["page_id"] for page in data.get("results", [])
                                      if "page_id" in page)
    seen = list(dict.fromkeys(retrieved))
    usage = result.usage
    return {
        "question": question, "mode": mode, "model": MODEL,
        "answer": str(result.output), "elapsed_seconds": round(elapsed, 3),
        "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "requests": usage.requests, "estimated_generation_usd": usage.cost,
        "tool_calls": calls, "retrieved_pages": seen, "reference_pages": reference,
        "reference_coverage": (len(set(reference) & set(seen)) / len(reference))
                              if reference else None,
    }


def compare(output_path: Path, limit: int) -> None:
    """同一質問・同一モデルの通常RAGと提案方式を順に実行し、実測値を保存する。"""
    from app.agent import agent as BASE_AGENT
    entries = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for case in CASES[:limit]:
        for label, target in (("vector", BASE_AGENT), ("hybrid", HYBRID_AGENT)):
            print(f"\n--- {case['name']} / {label} ---", flush=True)
            started = perf_counter()
            try:
                response = target.run_sync(case["question"], model=build_model())
                entry = _record(response, case["question"], label,
                                perf_counter() - started, case["reference_pages"])
                entries.append(entry)
                print(entry["answer"])
                print(f"[time] {entry['elapsed_seconds']}秒 / [usage] "
                      f"{entry['input_tokens']}/{entry['output_tokens']} tokens / "
                      f"[cost] {entry['estimated_generation_usd']} USD")
                print(f"[pages] {', '.join(entry['retrieved_pages'])}")
            except (ModelHTTPError, ValueError, OSError) as error:
                entries.append({"question": case["question"], "mode": label,
                                "error": str(error)})
                print(f"[error] {error}")
            finally:
                output_path.write_text(
                    json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
    print(f"\n比較結果: {output_path}（実測値。回答と引用の正誤は別途確認）")


def main() -> None:
    parser = argparse.ArgumentParser(description="第5回: Tree + Graph + ベクトル検索")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--compare", action="store_true", help="通常RAGと同一質問で比較する")
    parser.add_argument("--limit", type=int, choices=(1, 2, 3), default=3,
                        help="比較に使う質問数。既定3問")
    parser.add_argument("--output", type=Path, default=Path(".cache/rag/lec05-comparison.json"))
    parser.add_argument("--trace", action="store_true", help="ツール呼び出しを表示する")
    args = parser.parse_args()
    if args.compare:
        compare(args.output, args.limit)
        return
    if not args.question:
        parser.error("質問か--compareを指定してください")
    started = perf_counter()
    answer = HYBRID_AGENT.run_sync(args.question, model=build_model())
    print(answer.output)
    if args.trace:
        from app.agent import print_trace
        print_trace(answer, perf_counter() - started)


if __name__ == "__main__":
    main()
