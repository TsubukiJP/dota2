# AI翻訳ワークフロー(実験的)

このリポジトリには、英語→日本語の自動翻訳をPR化するGitHub Actionsを用意しています。

## 概要
- Issue起動: `.github/workflows/translate_on_issue.yml`
  - Issue本文の“自然言語プロンプト”を解析し、対象キーを抽出してAI翻訳→PR作成。
- 自動差分: `.github/workflows/translate_on_update.yml`
  - `main/resource/localization/abilities_english.txt.json` の更新検知時に差分キーのみ翻訳→PR作成。

## 前提
- リポジトリ Secrets に `OPENAI_API_KEY` を登録。
  - 既定モデルは `gpt-4o-mini`。`OPENAI_MODEL` で上書き可。

## Issueの書き方（自然言語OK）
- 例1: 「abilities_english.txt.json の line 120000-121000 を翻訳して」
- 例2: 「キーに "OraclesChallenge" を含むものをすべて翻訳して」
- 例3: 「Nemestice に関連するテキストを全部翻訳して。dota_english 優先」

解析結果は `resolved_targets.json` に、抽出されたキーは `keys.txt` に出力されます。0件の場合はPRを作らず、Issueに結果コメントを返します。

## 主要スクリプト
- `scripts/python/issue_prompt_resolver.py`
  - Issue本文から対象キーを推定（行レンジ/キー部分一致/トークン検索 + スコア）。
  - 出力: `keys.txt`, `resolved_targets.json`。
- `scripts/python/translate_diff_ai.py`
  - AI翻訳適用。`--english-auto --japanese-auto` 指定で `resolved_targets.json` を読んでファイルごとに自動処理。
  - プレースホルダ保持・数値差分優先・結果は `translation_changes.json` に出力。

## よくある質問
- APIキー未設定時は？
  - ワークフローはエラー終了し、ログに理由を出力します（Secretsに登録してください）。
- 既存訳がある場合は？
  - 数値/プレースホルダの変化がなければ維持、差分があれば更新提案に置き換えます。
