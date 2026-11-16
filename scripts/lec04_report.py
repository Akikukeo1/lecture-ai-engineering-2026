"""第4回実装課題: 人手ラベルとLLM採点の結果を再現可能な表にする。

実行:
  uv run python -m scripts.lec04_report \
    --cases data/eval/my_cases.jsonl \
    --labels lectures/04/my_labels.jsonl \
    --judge lectures/04/my_judge_result.json \
    --out lectures/04/my_report.md

採点済みの結果を集計するだけ。API呼び出しや自動ラベル生成はしない。
"""
import argparse
import json
from pathlib import Path

CRITERIA = ("grounding", "scope", "hint", "clarity", "lecture_mark")
NAMES = ("根拠", "範囲", "ヒント", "分かりやすさ", "回の表示")
VALID = {"pass", "fail"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze(cases: list[dict], labels: list[dict], verdicts: dict) -> dict:
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("ケースIDが重複しています")
    label_map = {label["id"]: label for label in labels}
    if len(label_map) != len(labels):
        raise ValueError("ラベルIDが重複しています")
    if set(ids) != set(label_map) or set(ids) != set(verdicts):
        raise ValueError("評価ケース・人手ラベル・LLM採点のID集合が一致しません")
    for case in cases:
        label = label_map[case["id"]]
        if "question" in label and label["question"] != case["question"]:
            raise ValueError(f"#{case['id']}: 評価ケースと人手ラベルの質問が一致しません")

    metrics = {}
    disagreements = []
    for key in CRITERIA:
        tp = fn = fp = tn = unsure = 0
        for case in cases:
            cid = case["id"]
            human = label_map[cid].get(key)
            automated = verdicts[cid][key]["verdict"]
            if human not in VALID:
                raise ValueError(f"{cid}/{key}: 人手ラベルが未確定です")
            if automated == "unsure":
                unsure += 1
                continue
            if automated not in VALID:
                raise ValueError(f"{cid}/{key}: 不明な判定 {automated}")
            if human == "fail":
                if automated == "fail":
                    tp += 1
                else:
                    fn += 1  # 不合格の見逃し
            elif automated == "fail":
                fp += 1
            else:
                tn += 1
            if automated != human:
                disagreements.append({
                    "id": cid, "criterion": key,
                    "kind": "見逃し" if human == "fail" else "過検出",
                    "reason": verdicts[cid][key]["reason"],
                })
        total = tp + fn + fp + tn + unsure
        metrics[key] = {
            "total": total, "compared": total - unsure, "agree": tp + tn,
            "unsure": unsure, "human_fail": tp + fn,
            "caught_fail": tp, "missed_fail": fn, "false_alarm": fp,
        }
    return {"n": len(cases), "metrics": metrics, "disagreements": disagreements}


def fraction(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "算出不可（分母0）"


def markdown(result: dict, model: str) -> str:
    n = result["n"]
    lines = [
        "# 第4回 独自評価セットの採点結果",
        "",
        f"- ケース数: {n}",
        f"- 採点モデル: `{model}`",
        "- 評価対象: 人手ラベルと採点器の一致（エージェントの正答率ではない）",
        "- `unsure` は一致率・不合格再現率の分母から除外し、件数を別掲する",
        "",
        "| 基準 | 一致率 | 判定可能率 | 要確認 | 不合格再現率 | 見逃し | 過検出 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, name in zip(CRITERIA, NAMES):
        m = result["metrics"][key]
        lines.append(
            f"| {name} | {fraction(m['agree'], m['compared'])} | "
            f"{fraction(m['compared'], n)} | {m['unsure']} | "
            f"{fraction(m['caught_fail'], m['human_fail'])} | "
            f"{m['missed_fail']} | {m['false_alarm']} |"
        )
    lines.extend(["", "## 不一致の内訳", ""])
    if not result["disagreements"]:
        lines.append("判定可能なケースの不一致はありません。")
    else:
        for d in result["disagreements"]:
            lines.append(f"- #{d['id']} / {d['criterion']} / {d['kind']}: {d['reason']}")
    lines.extend([
        "", "## 解釈上の制限",
        "",
        "- 10問では推定精度が低いため、結果を一般化しない。",
        "- 不合格の実例が0件なら再現率を100%とせず『算出不可』とする。",
        "- コードで採点する「回の表示」は形式の検査であり、引用内容の正確性を保証しない。",
        "- 現在の app.agent は第1〜3回資料のRAGを使うが、既存の rubric.md は用語集に基づく根拠評価である。両者の差に注意し、資料への忠実性は別途人手で確認する。",
        "- 人手ラベルは作成者による1人の判定であり、独立した複数人の合意ではない。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="第4回実装課題の採点品質を集計する")
    parser.add_argument("--cases", type=Path, default=Path("data/eval/my_cases.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("lectures/04/my_labels.jsonl"))
    parser.add_argument("--judge", type=Path, default=Path("lectures/04/my_judge_result.json"))
    parser.add_argument("--out", type=Path, default=Path("lectures/04/my_report.md"))
    args = parser.parse_args()
    data = json.loads(args.judge.read_text(encoding="utf-8"))
    if data.get("model") == "test":
        raise SystemExit("偽モデルの結果は提出用レポートにできません。実際のLLMを実行してください")
    result = analyze(load_jsonl(args.cases), load_jsonl(args.labels), data["verdicts"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown(result, data["model"]), encoding="utf-8")
    print(f"レポート生成: {args.out}")


if __name__ == "__main__":
    main()
