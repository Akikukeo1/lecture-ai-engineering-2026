"""第1回 発展課題: モデルと質問による応答時間・トークン数の比較。

実行例:
    uv run python -m app.compare_models \
      --models gemini-3.5-flash-lite gemini-3.6-flash \
      --output results/lec01-comparison.csv

無料枠に配慮してAPI呼び出しの間隔を設け、429は記録して停止する。
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from app.config import get_api_key

DEFAULT_QUESTIONS = (
    "LoRAとは何ですか？",
    "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
)
FIELDS = (
    "model",
    "question",
    "trial",
    "elapsed_seconds",
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
    "status",
    "response_text",
)


def measure(
    client: Any,
    models: Sequence[str],
    questions: Sequence[str],
    *,
    trials: int = 1,
    delay: float = 3.0,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    api_error_type: type[Exception],
) -> list[dict[str, Any]]:
    """モデル×質問×試行の結果を返す。429が出たら追加呼び出ししない。"""
    if trials < 1 or delay < 0:
        raise ValueError("trials は1以上、delayは0以上にしてください")

    tasks = [(model, question, trial) for model in models for question in questions for trial in range(1, trials + 1)]
    rows: list[dict[str, Any]] = []
    for index, (model, question, trial) in enumerate(tasks):
        if index and delay:
            sleep(delay)
        started = clock()
        response = None
        status = "ok"
        try:
            response = client.models.generate_content(model=model, contents=question)
        except api_error_type as exc:
            code = getattr(exc, "code", None)
            status = f"http_{code}" if code is not None else "api_error"
        elapsed = clock() - started
        usage = getattr(response, "usage_metadata", None)
        rows.append(
            {
                "model": model,
                "question": question,
                "trial": trial,
                "elapsed_seconds": round(elapsed, 3),
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
                "status": status,
                "response_text": (getattr(response, "text", None) or ""),
            }
        )
        if status == "http_429":
            break
    return rows


def save_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="第1回 発展課題: Geminiモデルの比較")
    parser.add_argument("--models", nargs="+", required=True, help="比較するモデルIDを2種類以上")
    parser.add_argument("--question", action="append", dest="questions", help="質問（複数回指定可）")
    parser.add_argument("--trials", type=int, default=1, help="各モデル・質問の試行回数（既定: 1）")
    parser.add_argument("--delay", type=float, default=3.0, help="呼び出し間の待機秒数（既定: 3）")
    parser.add_argument("--output", type=Path, default=Path("results/lec01-comparison.csv"))
    args = parser.parse_args()
    if len(set(args.models)) < 2:
        parser.error("異なるモデルを2種類以上指定してください")
    if args.trials < 1 or args.delay < 0:
        parser.error("trialsは1以上、delayは0以上にしてください")

    # import時にはAPIキーやSDKは不要。テストはモックで実行できる。
    from google import genai
    from google.genai import errors

    client = genai.Client(api_key=get_api_key())
    try:
        rows = measure(
            client,
            args.models,
            args.questions or DEFAULT_QUESTIONS,
            trials=args.trials,
            delay=args.delay,
            api_error_type=errors.APIError,
        )
    finally:
        client.close()

    save_csv(rows, args.output)
    for row in rows:
        print(
            f"[{row['status']}] {row['model']} / 試行{row['trial']} / "
            f"{row['elapsed_seconds']:.3f}秒 / "
            f"入力{row['prompt_tokens']} / 出力{row['output_tokens']} / 合計{row['total_tokens']} tokens"
        )
    print(f"CSV: {args.output} ({len(rows)}件)")
    if rows and rows[-1]["status"] == "http_429":
        print("429（レート制限）を検出したため、追加のリクエストは実行していません。")


if __name__ == "__main__":
    main()
