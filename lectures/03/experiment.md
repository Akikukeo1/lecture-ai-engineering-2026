# 第3回発展課題：Pydantic AI / ADK 比較実験

## 目的

第3回で扱った「モデル＋ループ＋ツール」のうち、ループを実装する
フレームワークだけを変更したときの挙動を観察する。

Pydantic AI と Google ADK に同一の `INSTRUCTIONS`、同一の Python 関数
（`search_glossary`、`get_lecture_summary`、`make_quiz`）、
同一の `AIE_MODEL` を渡す。質問は4カテゴリ各5問の20問。

## 実行

リポジトリ直下で、環境変数 `GEMINI_API_KEY` 設定済みの環境から次を実行する。

```bash
PYTHONPATH=. uv run --with 'google-adk>=2,<3' python -m scripts.compare_agent_frameworks
```

実測ファイル `results/lec03_frameworks.json` をそのまま共有すれば、
回答本文・ツール呼び出し・モデル呼び出し回数・実行時間を後から分析できる。
APIキーはファイルに出力しない。途中で止まっても実行済み結果は逐次保存する。
事前の動作確認だけなら `--limit 2 --repeats 1` で4試行に絞れる。
通常は20問×3回×2方式で120試行。レート制限に注意。

## 評価軸

- 指定ツール使用率：各質問で期待するツール名の呼び出しが1回以上あった比率
- 平均ツール呼び出し回数、平均モデル呼び出し回数、平均処理時間
- 個別回答：用語集と講義要点に照らして、正確性、根拠のない断定を**別途**評価する
- 特に「用語集にないこと」を問うケースでは検索しただけでは成功としない

**注意：**指定ツール使用率は正答率ではない。
モデルやツールスキーマ、フレームワーク内部の既定動作は完全には同一でないため、
差があっても「フレームワークそのものが原因」とは断定できない。
小標本・時間順のAPI負荷・モデル非決定性の影響を含む探索的比較である。
## 初回実験と再測定

2026-11-09に60試行ずつ実行。ADK版60/60件のツール呼び出し等を記録できた。
Pydantic AI版60件はランナーの `result.usage()` が現行APIと不一致
（`RunUsage` は呼び出し可能ではなくプロパティ）で、計測が失敗した。
`answer.md` にこの測定上の制約とADKの定量・定性結果を記録した。

元のJSONをCodespaces内に残した状態で、Pydantic側だけを再実行する:

```bash
cp results/lec03_frameworks.json results/lec03_frameworks_before_fix.json
PYTHONPATH=. uv run --with 'google-adk>=2,<3' python -m scripts.compare_agent_frameworks --repair-pydantic
```

`--repair-pydantic` は既存JSONの60件のADK実測を残し、
同じ20問×3回のPydantic AI版60試行を再実行する。
`GEMINI_API_KEY` と `AIE_MODEL`（設定している場合）は以前と同じ値を使用する。
集計バグ修正後でもAPIの混雑や時間帯の影響が残るため、遅延時間の厳密な優劣は主張しない。
また、「基礎・応用・発展か」という誘導質問へのADK回答は3件中2件で
「はい」と肯定した一方、続く列挙では「基礎・実装・発展」を正しく示し、
検索実施と正確性が独立していることを確認した。
