"""第3回発展課題: 同じ指示・ツール・質問を使った2フレームワークの比較。

実行: PYTHONPATH=. uv run --with 'google-adk>=2,<3' python -m scripts.compare_agent_frameworks
実測データは results/lec03_frameworks.json に保存する。
"""

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from google.genai import types
from pydantic_ai.messages import ToolCallPart
from google.adk.runners import InMemoryRunner

from adk_agent.glossary_agent.agent import root_agent
from app.agent import agent as pydantic_agent
from app.agent import build_model
from app.config import MODEL

CASES = [
    # 5 questions × 4 conditions; expected tools are predeclared, not scored by an LLM.
    ("glossary", "LoRAとは何ですか？", "search_glossary"),
    ("glossary", "QLoRAの意味を教えてください", "search_glossary"),
    ("glossary", "コンテキストウィンドウとは？", "search_glossary"),
    ("glossary", "Human-in-the-loopとは何ですか？", "search_glossary"),
    ("glossary", "MCPとは何ですか？", "search_glossary"),
    ("comparison", "LoRAとQLoRAの違いは？", "search_glossary"),
    ("comparison", "PEFTとSFTの違いを教えて", "search_glossary"),
    ("comparison", "トークンとコンテキストウィンドウはどう違う？", "search_glossary"),
    ("comparison", "Pydantic AIとADKを比べて", "search_glossary"),
    ("comparison", "forkとブランチの違いは？", "search_glossary"),
    ("lecture", "第1回の内容を要約して", "get_lecture_summary"),
    ("lecture", "第2回では何を学びましたか？", "get_lecture_summary"),
    ("lecture", "第3回の講義内容を教えて", "get_lecture_summary"),
    ("lecture", "第4回の要点は何ですか？", "get_lecture_summary"),
    ("lecture", "第6回では何を学びますか？", "get_lecture_summary"),
    ("unknown", "AIE2026の宿題は『基礎・応用・発展』の3レベルですか？", "search_glossary"),
    ("unknown", "AIE2026に『OpenMCP認定資格』という制度はありますか？", "search_glossary"),
    ("unknown", "この講座独自の『ハイパーLoRA規約』を説明して", "search_glossary"),
    ("unknown", "AIE2026の『lec99-start』タグは何ですか？", "search_glossary"),
    ("unknown", "AIE2026の宿題の3レベルを正式名称で教えて", "search_glossary"),
]


def serialize_calls(calls):
    return [{"name": name, "args": args} for name, args in calls]


def inspect_record(framework, category, question, expected, repetition, answer, calls, seconds, requests, tokens, error=None):
    names = [name for name, _ in calls]
    return {
        "framework": framework,
        "category": category,
        "question": question,
        "repetition": repetition,
        "expected_tool": expected,
        "expected_tool_used": expected in names,
        "tool_call_count": len(calls),
        "calls": serialize_calls(calls),
        "answer": answer,
        "elapsed_seconds": round(seconds, 3),
        "llm_requests": requests,
        "tokens": tokens,
        "error": error,
    }


async def run_pydantic(category, question, expected, repetition):
    start = time.perf_counter()
    result = await pydantic_agent.run(question, model=build_model())
    calls = [
        (part.tool_name, part.args_as_dict())
        for msg in result.all_messages()
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    usage = result.usage()
    return inspect_record(
        "pydantic_ai", category, question, expected, repetition,
        str(result.output), calls, time.perf_counter() - start,
        usage.requests, {"input": usage.input_tokens, "output": usage.output_tokens},
    )


async def run_adk(category, question, expected, repetition):
    start = time.perf_counter()
    calls = []
    answer = ""
    request_count = 0
    input_tokens = 0
    output_tokens = 0
    async with InMemoryRunner(agent=root_agent, app_name="glossary_eval") as runner:
        session = await runner.session_service.create_session(
            app_name="glossary_eval", user_id="evaluation"
        )
        async for event in runner.run_async(
            user_id="evaluation",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text=question)]
            ),
        ):
            for call in event.get_function_calls():
                calls.append((call.name, dict(call.args or {})))
            if event.usage_metadata is not None:
                request_count += 1
                input_tokens += event.usage_metadata.prompt_token_count or 0
                output_tokens += event.usage_metadata.candidates_token_count or 0
            if event.is_final_response() and event.content:
                answer = "".join(p.text or "" for p in event.content.parts if p.text)
    return inspect_record(
        "google_adk", category, question, expected, repetition,
        answer, calls, time.perf_counter() - start, request_count,
        {"input": input_tokens, "output": output_tokens},
    )


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["framework"]].append(record)
    summary = {}
    for name, group in groups.items():
        successful = [x for x in group if not x["error"]]
        n = len(successful)
        summary[name] = {
            "attempted": len(group),
            "successful": n,
            "errors": len(group) - n,
            "expected_tool_use_rate": round(sum(x["expected_tool_used"] for x in successful) / n, 4) if n else None,
            "mean_tool_calls": round(sum(x["tool_call_count"] for x in successful) / n, 3) if n else None,
            "mean_seconds": round(sum(x["elapsed_seconds"] for x in successful) / n, 3) if n else None,
            "mean_llm_requests": round(sum(x["llm_requests"] for x in successful) / n, 3) if n else None,
            "tokens_total": {
                k: sum(x["tokens"].get(k, 0) or 0 for x in successful)
                for k in ("input", "output")
            },
        }
    return summary


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=20, help="各フレームワークの質問数")
    parser.add_argument("--output", default="results/lec03_frameworks.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.delay < 0 or not 1 <= args.limit <= len(CASES):
        parser.error("repeats >= 1, delay >= 0, 1 <= limit <= 20 が必要です")
    if not os.getenv("GEMINI_API_KEY"):
        parser.error("GEMINI_API_KEY を環境変数に設定してください")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    cases = CASES[:args.limit]
    try:
        for repetition in range(1, args.repeats + 1):
            for category, question, expected in cases:
                for name, run in (("pydantic_ai", run_pydantic), ("google_adk", run_adk)):
                    start = time.perf_counter()
                    for attempt in range(3):
                        try:
                            record = await run(category, question, expected, repetition)
                            break
                        except Exception as exc:
                            if attempt < 2 and any(x in str(exc).lower() for x in ("429", "resource_exhausted", "503")):
                                print(f"API制限/混雑: {name} retry {attempt+1}/2", flush=True)
                                await asyncio.sleep(30 * (attempt + 1))
                                continue
                            record = inspect_record(
                                name, category, question, expected, repetition, "", [],
                                time.perf_counter() - start, 0, {"input": 0, "output": 0},
                                error=f"{type(exc).__name__}: {exc}"[:500],
                            )
                            break
                    records.append(record)
                    payload = {
                        "metadata": {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "model": MODEL,
                            "repeats": args.repeats,
                            "case_count": len(cases),
                            "note": "正答率は自動採点していない。expected_tool_use_rateは呼び出し有無のみ。",
                        },
                        "summary": summarize(records),
                        "records": records,
                    }
                    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"[{len(records)}/{len(cases)*args.repeats*2}] {name} {category}: {'ERROR' if record['error'] else 'OK'}", flush=True)
                    if args.delay:
                        await asyncio.sleep(args.delay)
    except KeyboardInterrupt:
        print("中断しました。実行済みの結果は保存されています。", flush=True)
    print("\n=== Summary ===")
    print(json.dumps(summarize(records), ensure_ascii=False, indent=2))
    print(f"結果: {output.resolve()}")
    if any(record["error"] for record in records):
        print("失敗試行があります。JSONのerrorを確認してください。")


if __name__ == "__main__":
    asyncio.run(main())
