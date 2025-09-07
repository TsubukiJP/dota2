#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, sys
from pathlib import Path
import yaml

PH_RE = re.compile(r"(%s\d+|%d\w*|%f\w*|%\w+%|\{[sg]:[^}]+\}|<[^>]+>)")
NUM_RE = re.compile(r"\d+")

# 軽い正規化（比較用）
def normalize(s: str) -> str:
    if s is None:
        return ""
    # 半角/全角の%は比較時には同一視、空白も丸める
    s = s.replace('%', '％')
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# 置換を安全に行うため、プレースホルダとHTMLを一旦退避
class Placeholders:
    def __init__(self):
        self.bucket = []
    def hide(self, text: str) -> str:
        def _repl(m):
            idx = len(self.bucket)
            self.bucket.append(m.group(0))
            return f"__PH_{idx}__"
        return PH_RE.sub(_repl, text)
    def show(self, text: str) -> str:
        for i, val in enumerate(self.bucket):
            text = text.replace(f"__PH_{i}__", val)
        return text

# 単位などの後処理
UNITS = [
    (re.compile(r"(\b\d+)\s+seconds\b"), r"\1秒"),
    (re.compile(r"(\b\d+)\s+second\b"), r"\1秒"),
]

# 置換（英→日）を長い順で適用
class Replacer:
    def __init__(self, replacements: dict):
        # 長いキーから順に置換
        self.items = sorted(replacements.items(), key=lambda kv: len(kv[0]), reverse=True)
    def apply(self, text: str) -> str:
        for en, ja in self.items:
            text = text.replace(en, ja)
        return text

# 既存JPと新訳を比較して更新要否と理由を返す

def should_update(old_jp: str, new_jp: str):
    if old_jp is None:
        return True, 'added'
    if normalize(old_jp) == normalize(new_jp):
        return False, 'no-change'
    # 数字の変化・プレースホルダの変化を優先理由に
    old_nums = NUM_RE.findall(old_jp)
    new_nums = NUM_RE.findall(new_jp)
    if old_nums != new_nums:
        return True, 'number_change'
    old_ph = PH_RE.findall(old_jp)
    new_ph = PH_RE.findall(new_jp)
    if old_ph != new_ph:
        return True, 'placeholder_change'
    return True, 'text_change'

# ルール読込 + 既存辞書から対訳を自動抽出（ショートフレーズのみ）

def build_replacements(rules: dict, en_map: dict, ja_map: dict) -> dict:
    rep = dict(rules.get('replacements', {}))
    # 既存の短い定訳をペアリング
    for k, en_val in en_map.items():
        ja_val = ja_map.get(k)
        if not ja_val:
            continue
        # 短い用語（UI的な単語群）だけ自動登録
        if 1 <= len(en_val) <= 40 and '\n' not in en_val and '<' not in en_val and '{' not in en_val:
            # 日本語側も短文でタグのないもの
            if len(ja_val) <= 40 and '<' not in ja_val and '{' not in ja_val:
                rep.setdefault(en_val, ja_val)
    return rep

# 翻訳本体

def translate_text(en_text: str, replacer: Replacer) -> str:
    ph = Placeholders()
    tmp = ph.hide(en_text)
    tmp = replacer.apply(tmp)
    # 単位変換
    for rx, repl in UNITS:
        tmp = rx.sub(repl, tmp)
    # 仕上げ
    tmp = re.sub(r"\s+", " ", tmp).strip()
    return ph.show(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--english', required=True)
    ap.add_argument('--japanese', required=True)
    ap.add_argument('--rules', required=True)
    ap.add_argument('--keys')
    ap.add_argument('--keys-file')
    args = ap.parse_args()

    en_path = Path(args.english)
    ja_path = Path(args.japanese)
    rules_path = Path(args.rules)

    if not en_path.exists() or not ja_path.exists():
        print('Input files not found', file=sys.stderr)
        sys.exit(2)

    # 対象キー
    keys = []
    if args.keys:
        # 改行・カンマ区切り両対応
        for part in re.split(r"[\n,]", args.keys):
            part = part.strip()
            if part:
                keys.append(part)
    if args.keys_file:
        p = Path(args.keys_file)
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line:
                    keys.append(line)
    if not keys:
        print('No keys provided. Use --keys or --keys-file.', file=sys.stderr)
        sys.exit(2)

    en = json.loads(en_path.read_text(encoding='utf-8'))
    ja = json.loads(ja_path.read_text(encoding='utf-8'))

    rules = yaml.safe_load(rules_path.read_text(encoding='utf-8')) or {}
    replacements = build_replacements(rules, en, ja)
    replacer = Replacer(replacements)

    changes = []
    updated = False

    for key in keys:
        en_val = en.get(key)
        if en_val is None:
            changes.append({
                'key': key,
                'action': 'skipped',
                'reason': 'missing_in_english'
            })
            continue
        # 文字列以外はスキップ
        if not isinstance(en_val, str):
            changes.append({'key': key, 'action': 'skipped', 'reason': 'non_string'})
            continue

        new_jp = translate_text(en_val, replacer)
        old_jp = ja.get(key)
        do_update, reason = should_update(old_jp, new_jp)
        if do_update:
            ja[key] = new_jp
            updated = True
            changes.append({'key': key, 'action': 'updated' if old_jp is not None else 'added', 'reason': reason, 'en': en_val, 'old_jp': old_jp, 'new_jp': new_jp})
        else:
            changes.append({'key': key, 'action': 'kept', 'reason': reason, 'en': en_val, 'old_jp': old_jp})

    # 結果を書き出し
    Path('translation_changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf-8')

    if updated:
        ja_path.write_text(json.dumps(ja, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
        print('Japanese file updated.')
    else:
        print('No changes were necessary.')

if __name__ == '__main__':
    main()