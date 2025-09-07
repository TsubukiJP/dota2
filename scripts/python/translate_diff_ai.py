#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
英→日のAI翻訳で差分のみを更新するスクリプト。

機能:
- Gitのコミット範囲から変更されたキーを抽出、または --keys/--keys-file/正規表現で指定
- 用語集(glossary.csv)による固定訳/英語固定をプロンプトに反映
- プレースホルダ(%s, %1$s, {s:*}, <font …>)保護
- 既存訳との差分判定（数値/プレースホルダ変化を優先）
- translation_changes.json 形式で結果を出力

拡張(イシュー連携):
- --english-auto/--japanese-auto で resolved_targets.json を読み、ファイルごとに自動実行
- その場合は各ファイルのキー群を自動でバッチ処理し、変更ログを集約

依存: Python 3.9+, requests, pyyaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import requests  # type: ignore
except Exception:
    requests = None


# プレースホルダ
PH_RE = re.compile(r"(%\d+\$[sdf]|%[sdf]|%[A-Z_]+%|\{[a-z]:[^}]+\}|<[^>]+>)")
NUM_RE = re.compile(r"\d+")


def eprint(*args):
    print(*args, file=sys.stderr)


def normalize(s: Optional[str]) -> str:
    if s is None:
        return ""
    t = re.sub(r"\s+", " ", s)
    return t.strip()


class Placeholders:
    def __init__(self) -> None:
        self.bucket: List[str] = []

    def hide(self, text: str) -> str:
        def _repl(m: re.Match[str]) -> str:
            idx = len(self.bucket)
            self.bucket.append(m.group(0))
            return f"__PH_{idx}__"

        return PH_RE.sub(_repl, text)

    def show(self, text: str) -> str:
        for i, val in enumerate(self.bucket):
            text = text.replace(f"__PH_{i}__", val)
        return text

    def tokens_in(self, text: str) -> List[str]:
        return re.findall(r"__PH_\d+__", text)


def load_json_file(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def git_show(path: str, rev: str) -> Optional[str]:
    try:
        cp = subprocess.run([
            "git", "--no-pager", "show", f"{rev}:{path}"
        ], check=True, capture_output=True)
        return cp.stdout.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError:
        return None


def load_json_at_rev(path: str, rev: str) -> Dict[str, object]:
    content = git_show(path, rev)
    if content is None:
        return {}
    try:
        return json.loads(content)
    except Exception:
        return {}


def union_keys(a: Dict[str, object], b: Dict[str, object]) -> List[str]:
    return sorted(set(a.keys()) | set(b.keys()))


def changed_string_keys(base: Dict[str, object], head: Dict[str, object]) -> List[str]:
    keys: List[str] = []
    for k in union_keys(base, head):
        bv = base.get(k)
        hv = head.get(k)
        if hv is None:
            continue
        if not isinstance(hv, str):
            continue
        if bv is None:
            keys.append(k)
        elif isinstance(bv, str) and bv != hv:
            keys.append(k)
    return keys


def read_glossary(path: Optional[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path or not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader((line for line in f if not line.lstrip().startswith('#')))
        for r in reader:
            if not r:
                continue
            rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows


def glossary_hints(glossary: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
    keep: List[str] = []
    pairs: List[str] = []
    for r in glossary:
        en = r.get('en', '')
        ja = r.get('ja', '')
        policy = (r.get('policy', '') or '').lower()
        if not en:
            continue
        if policy == 'keep_en':
            keep.append(en)
        elif policy == 'translate' and ja:
            pairs.append(f"{en} => {ja}")
    return keep, pairs


def build_system_prompt(keep_terms: List[str], pairs: List[str]) -> str:
    lines = [
        "あなたはDota 2の用語に精通した、厳密な置換規則を守る日英翻訳者です。",
        "出力は日本語のみ。トークンやプレースホルダは絶対に改変しないでください。",
        "- プレースホルダ: %s, %d, %1$s, %MODIFIER_*%, {s:*}, {i:*}, <font …> はそのまま保持。",
        "- 数値・記号・改行は保持。",
        "- 文体: 常体。UIラベルは簡潔に。",
    ]
    if keep_terms:
        lines.append("- 以下の用語は英語表記固定: " + ", ".join(keep_terms[:40]) + (" …" if len(keep_terms) > 40 else ""))
    if pairs:
        lines.append("- 固定訳(必ずこの訳語):")
        for p in pairs[:80]:
            lines.append(f"  * {p}")
        if len(pairs) > 80:
            lines.append("  * …")
    return "\n".join(lines)


def call_openai(api_key: str, model: str, system_prompt: str, text: str) -> str:
    if requests is None:
        raise RuntimeError("requests が未インストールです。")
    url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                "次の英文を日本語へ。\n"
                "- プレースホルダ(__PH_0__ 等)は改変・追加・削除しない\n"
                "- 語順は自然な日本語に、冗長な助詞は省く\n"
                "- 出力は本文のみ\n\n" + text
            )},
        ],
    }
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    if resp.status_code >= 300:
        raise RuntimeError(f"OpenAI API error: {resp.status_code} {resp.text}")
    out = resp.json()
    try:
        return out["choices"][0]["message"]["content"].strip()
    except Exception:
        raise RuntimeError(f"Unexpected OpenAI response: {out}")


def should_update(old_jp: Optional[str], new_jp: str) -> Tuple[bool, str]:
    if old_jp is None:
        return True, 'added'
    if normalize(old_jp) == normalize(new_jp):
        return False, 'no-change'
    old_nums = NUM_RE.findall(old_jp)
    new_nums = NUM_RE.findall(new_jp)
    if old_nums != new_nums:
        return True, 'number_change'
    old_ph = PH_RE.findall(old_jp)
    new_ph = PH_RE.findall(new_jp)
    if old_ph != new_ph:
        return True, 'placeholder_change'
    return True, 'text_change'


def decide_targets(args, en_head: Dict[str, object], en_base: Dict[str, object]) -> List[str]:
    keys: List[str] = []
    if args.keys:
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
    if not keys and (args.filter_key or args.filter_value):
        key_rx = re.compile(args.filter_key) if args.filter_key else None
        val_rx = re.compile(args.filter_value) if args.filter_value else None
        for k, v in en_head.items():
            if not isinstance(v, str):
                continue
            if key_rx and not key_rx.search(k):
                continue
            if val_rx and not val_rx.search(v):
                continue
            keys.append(k)
    if not keys:
        keys = changed_string_keys(en_base, en_head)
    return sorted(set(keys))


def run_single(en_path: Path, ja_path: Path, keys: List[str], glossary: List[Dict[str, str]], api_key: str, model: str, changes_path: Path, dry_run: bool) -> Tuple[bool, List[Dict[str, object]]]:
    keep_terms, pair_terms = glossary_hints(glossary)
    system_prompt = build_system_prompt(keep_terms, pair_terms)
    use_ai = bool(api_key) and not dry_run

    en = load_json_file(en_path)
    ja = load_json_file(ja_path)

    changes: List[Dict[str, object]] = []
    updated = False

    for key in keys:
        en_val = en.get(key)
        if not isinstance(en_val, str):
            changes.append({'key': key, 'action': 'skipped', 'reason': 'non_string'})
            continue

        ph = Placeholders()
        hidden_en = ph.hide(en_val)
        try:
            jp_out = call_openai(api_key, model, system_prompt, hidden_en) if use_ai else hidden_en
        except Exception as ex:
            changes.append({'key': key, 'action': 'skipped', 'reason': f'api_error: {ex}', 'en': en_val})
            continue

        out_tokens = ph.tokens_in(jp_out)
        if len(set(out_tokens)) < len(ph.bucket):
            changes.append({'key': key, 'action': 'kept', 'reason': 'placeholder_mismatch', 'en': en_val, 'suggested_jp_hidden': jp_out})
            continue
        new_jp = ph.show(jp_out)

        old_jp = ja.get(key) if isinstance(ja.get(key), str) else None
        do_update, reason = should_update(old_jp, new_jp)
        if do_update:
            ja[key] = new_jp
            updated = True
            changes.append({'key': key, 'action': 'updated' if old_jp is not None else 'added', 'reason': reason, 'en': en_val, 'old_jp': old_jp, 'new_jp': new_jp})
        else:
            changes.append({'key': key, 'action': 'kept', 'reason': reason, 'en': en_val, 'old_jp': old_jp})

    changes_path.write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf-8')
    if updated:
        ja_path.write_text(json.dumps(ja, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    return updated, changes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--english')
    ap.add_argument('--japanese')
    ap.add_argument('--glossary', default='docs/glossary.csv')
    ap.add_argument('--commit-range')
    ap.add_argument('--base')
    ap.add_argument('--head')
    ap.add_argument('--keys')
    ap.add_argument('--keys-file')
    ap.add_argument('--filter-key', help='正規表現: key名フィルタ')
    ap.add_argument('--filter-value', help='正規表現: 英文値フィルタ')
    ap.add_argument('--openai-model', default=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--english-auto', action='store_true', help='resolved_targets.jsonから英語ファイルを推定')
    ap.add_argument('--japanese-auto', action='store_true', help='resolved_targets.jsonから日本語ファイルを推定')
    ap.add_argument('--resolved', default='resolved_targets.json')
    ap.add_argument('--changes-path', default='translation_changes.json')
    args = ap.parse_args()

    # APIキー
    api_key = os.getenv('OPENAI_API_KEY', '')
    if not api_key and not args.dry_run:
        eprint('OPENAI_API_KEY が未設定です。環境変数に設定してください。')
        Path(args.changes_path).write_text(json.dumps({'error': 'OPENAI_API_KEY missing'}), encoding='utf-8')
        sys.exit(2)

    glossary = read_glossary(Path(args.glossary))

    # オート解決: resolved_targets.json を用いてファイルごとに実行
    if args.english_auto or args.japanese_auto:
        res_p = Path(args.resolved)
        if not res_p.exists():
            eprint(f'resolved_targets.json が見つかりません: {res_p}')
            Path(args.changes_path).write_text(json.dumps({'error': 'resolved_targets.json missing'}), encoding='utf-8')
            sys.exit(2)
        resolved = json.loads(res_p.read_text(encoding='utf-8'))
        groups: Dict[str, List[str]] = {}
        for r in resolved.get('results', []):
            k = r.get('key'); f = r.get('file')
            if isinstance(k, str) and isinstance(f, str):
                groups.setdefault(f, []).append(k)
        if not groups:
            Path('keys.txt').write_text('', encoding='utf-8')
            Path(args.changes_path).write_text('[]', encoding='utf-8')
            print('No target keys from resolver.')
            return
        any_updated = False
        all_changes: List[Dict[str, object]] = []
        for en_file, klist in groups.items():
            en_path = Path(en_file)
            ja_path = Path(en_file.replace('_english.txt.json', '_japanese.txt.json'))
            if not en_path.exists() or not ja_path.exists():
                all_changes.append({'error': 'file_missing', 'english': str(en_path), 'japanese': str(ja_path)})
                continue
            updated, changes = run_single(en_path, ja_path, klist, glossary, api_key, args.openai_model, Path('translation_changes.json'), args.dry_run)
            any_updated = any_updated or updated
            if isinstance(changes, list):
                all_changes.extend(changes)
        Path(args.changes_path).write_text(json.dumps(all_changes, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Japanese file(s) updated.' if any_updated else 'No changes were necessary.')
        return

    # 通常モード: 1ファイル指定
    if not args.english or not args.japanese:
        eprint('english/japanese のパスを指定するか、--english-auto/--japanese-auto を使ってください。')
        Path(args.changes_path).write_text(json.dumps({'error': 'missing paths'}), encoding='utf-8')
        sys.exit(2)

    en_path = Path(args.english)
    ja_path = Path(args.japanese)
    if not en_path.exists() or not ja_path.exists():
        eprint('Input files not found.')
        Path(args.changes_path).write_text(json.dumps({'error': 'input files not found'}), encoding='utf-8')
        sys.exit(2)

    # base/head 解決
    base_rev = args.base
    head_rev = args.head
    if args.commit_range and ('..' in args.commit_range):
        base_rev, head_rev = args.commit_range.split('..', 1)
    if not head_rev:
        head_rev = 'HEAD'
    if not base_rev:
        base_rev = os.getenv('GITHUB_BASE_SHA') or os.getenv('BEFORE_SHA') or 'HEAD^'

    en_head = load_json_at_rev(str(en_path).replace('\\', '/'), head_rev)
    if not en_head:
        en_head = load_json_file(en_path)
    en_base = load_json_at_rev(str(en_path).replace('\\', '/'), base_rev)

    targets = decide_targets(args, en_head, en_base)
    if not targets:
        eprint('翻訳対象キーがありません。')
        Path(args.changes_path).write_text('[]', encoding='utf-8')
        print('No changes were necessary.')
        return

    updated, changes = run_single(en_path, ja_path, targets, glossary, api_key, args.openai_model, Path(args.changes_path), args.dry_run)
    print('Japanese file updated.' if updated else 'No changes were necessary.')


if __name__ == '__main__':
    main()

