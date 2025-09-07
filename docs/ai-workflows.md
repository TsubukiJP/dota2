# AI翻訳ワークフロー(実験的)

このリポジトリには、英語→日本語の自動翻訳をPR化するGitHub Actionsを追加しています。

## 概要
- `.github/workflows/translate_on_issue.yml`
  - Issue本文のYAMLで範囲を指定して、AI翻訳を実行しPRを作成します。
- `.github/workflows/translate_on_update.yml`
  - `main/resource/localization/abilities_english.txt.json` の更新検知時に差分キーのみ翻訳し、PRを作成します。

## 事前準備
- リポジトリの Secrets に `OPENAI_API_KEY` を追加。
  - 既定モデルは `gpt-4o-mini`。必要に応じて `OPENAI_MODEL` で上書き可能。

## Issueトリガの書式例
Issue本文に以下のようなYAMLブロックを入れてください。

```yaml
target: abilities  # dota / gameui なども可
# base/head を指定しない場合、keys か filter で範囲指定
keys:
  - DOTA_Tooltip_ability_foo
  - dota_ability_variable_cast_range
# もしくは正規表現フィルタ
# filter_key: "^DOTA_Tooltip_ability_"
# filter_value: "Cooldown|Mana Cost"
```

## スクリプトについて
- `scripts/python/translate_diff_ai.py`
  - 差分検出(コミット範囲/フィルタ/明示キー)→AI翻訳→JP更新を実施。
  - プレースホルダ(%s, %d, {s:*}, <font>等)は隠蔽→復元で厳密に保持。
  - 数値/プレースホルダの変化がある場合は更新を優先します。
  - 変更サマリを `translation_changes.json` に出力します。

## よくあるQ&A
- APIキーが無い場合は?
  - フォールバックとして原文を維持(または簡易置換)し、サマリのみ出力します。
- 既存訳がある場合は上書きされますか?
  - 数値/プレースホルダが変化していなければ、同一と判定される限り維持します。差分が大きい場合のみ更新します。

