#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Issue本文(自然言語)から翻訳対象キーを推定して keys.txt と resolved_targets.json を出力するツール。

対応:
- 行レンジ指定: "<file> の line A~B を翻訳"
- キー部分一致: "キーに \"Foo\" を含むもの"
- 自由記述(イベント名など): トークン抽出 + スコアリング(完全一致/前方一致/部分一致)

使い方:
  python scripts/python/issue_prompt_resolver.py \
    --prompt-file issue_prompt.txt \
    --english-paths "main/resource/localization/abilities_english.txt.json,main/resource/localization/dota_english.txt.json,main/resource/localization/gameui_english.txt.json" \
    --max-keys 200

出力:
- keys.txt: 1行1キー
- resolved_targets.json: ログ/根拠/スコア/対象ファイルなど
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable


KEY_RE = re.compile(r"^\s*\"([^\"]+)\"\s*:\s*")


def read_text(p: Path) -> str:
    return p.read_text(encoding='utf-8', errors='replace')


def load_json(p: Path) -> Dict[str, object]:
    return json.loads(p.read_text(encoding='utf-8'))


def detect_case_insensitive(prompt: str) -> bool:
    kw = ['ci', 'case-insensitive', '大小無視', '大文字小文字を無視']
    return any(k.lower() in prompt.lower() for k in kw)


def detect_target_files(prompt: str, english_paths: List[str]) -> List[str]:
    # 明示ファイル名/カテゴリヒント
    hints = []
    if re.search(r"abilities_english\.txt\.json", prompt, re.I):
        hints.append('abilities')
    if re.search(r"dota_english\.txt\.json", prompt, re.I):
        hints.append('dota')
    if re.search(r"gameui_english\.txt\.json", prompt, re.I):
        hints.append('gameui')
    if re.search(r"\babilities\b", prompt, re.I):
        hints.append('abilities')
    if re.search(r"\bdota\b", prompt, re.I):
        hints.append('dota')
    if re.search(r"\bgameui\b", prompt, re.I):
        hints.append('gameui')

    if hints:
        chosen = []
        for h in ['abilities', 'dota', 'gameui']:
            if h in hints:
                for p in english_paths:
                    if f"{h}_english.txt.json" in p:
                        chosen.append(p)
        return chosen
    return english_paths


def parse_line_ranges(prompt: str) -> List[Tuple[str, int, int]]:
    """return list of (maybe-file-hint, start, end)
    maybe-file-hint: may be a filename or category token; can be '' when not given
    """
    res: List[Tuple[str, int, int]] = []
    # <file> の line A~B
    for m in re.finditer(r"([A-Za-z0-9_./\\]*english\.txt\.json)\D*line\s*(\d+)\s*[\-~〜]\s*(\d+)", prompt, re.I):
        res.append((m.group(1), int(m.group(2)), int(m.group(3))))
    # abilities の line A-B 等
    for m in re.finditer(r"\b(abilities|dota|gameui)\b\D*line\s*(\d+)\s*[\-~〜]\s*(\d+)", prompt, re.I):
        res.append((m.group(1), int(m.group(2)), int(m.group(3))))
    # line A-B (ファイル未指定)
    for m in re.finditer(r"\bline\s*(\d+)\s*[\-~〜]\s*(\d+)", prompt, re.I):
        res.append(("", int(m.group(1)), int(m.group(2))))
    return res


def extract_keys_from_line_range(path: Path, start: int, end: int) -> Tuple[List[str], List[int]]:
    keys: List[str] = []
    hits_lines: List[int] = []
    with path.open(encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f, start=1):
            if i < start:
                continue
            if i > end:
                break
            m = KEY_RE.match(line)
            if m:
                keys.append(m.group(1))
                hits_lines.append(i)
    return keys, hits_lines


def list_all_keys(en_map: Dict[str, object]) -> Iterable[str]:
    for k, v in en_map.items():
        if isinstance(v, (str, int, float, bool)):
            yield k


def gather_key_contains(en_map: Dict[str, object], substrings: List[str], ci: bool) -> List[str]:
    keys: List[str] = []
    for k in list_all_keys(en_map):
        for sub in substrings:
            if ci:
                if sub.lower() in k.lower():
                    keys.append(k)
                    break
            else:
                if sub in k:
                    keys.append(k)
                    break
    return keys


def quoted_tokens(prompt: str) -> List[str]:
    toks = re.findall(r'"([^"]{2,})"', prompt)
    return list(dict.fromkeys(toks))


def camel_snake_tokens(prompt: str) -> List[str]:
    toks = []
    toks += re.findall(r"[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+", prompt)
    toks += re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", prompt)
    toks += re.findall(r"[\u3040-\u30FF\u4E00-\u9FFF]{2,}", prompt)
    # unique preserve order
    return list(dict.fromkeys(toks))


def score_hit(s: str, tok: str, ci: bool) -> Tuple[int, bool, bool]:
    if ci:
        S = s.lower(); T = tok.lower()
    else:
        S = s; T = tok
    exact = (S == T)
    prefix = S.startswith(T)
    substr = (T in S)
    score = 0
    if exact:
        score = 3
    elif prefix:
        score = 2
    elif substr:
        score = 1
    else:
        # fuzzy補助
        r = difflib.SequenceMatcher(None, S, T).ratio()
        if r >= 0.86:
            score = 1
    return score, exact, substr or prefix or exact


def fuzzy_collect(en_map: Dict[str, object], tokens: List[str], ci: bool) -> List[Tuple[str, int, str, List[str]]]:
    """return list of (key, score, hit_type, matched_tokens)"""
    out: List[Tuple[str, int, str, List[str]]] = []
    for k, v in en_map.items():
        if not isinstance(v, str):
            continue
        key_score = 0
        val_score = 0
        matched_k: List[str] = []
        matched_v: List[str] = []
        for t in tokens:
            s, _, hit = score_hit(k, t, ci)
            key_score += s
            if hit:
                matched_k.append(t)
            s2, _, hit2 = score_hit(v, t, ci)
            val_score += s2
            if hit2:
                matched_v.append(t)
        score = max(key_score, val_score)
        if key_score > 0:
            score += 1  # キー命中ボーナス
        # 全トークン命中ボーナス
        if set(tokens) and (set(tokens).issubset(set(matched_k)) or set(tokens).issubset(set(matched_v))):
            score += 1
        if score >= 2:
            hit_type = 'both' if (key_score > 0 and val_score > 0) else ('key' if key_score > 0 else 'value')
            out.append((k, score, hit_type, list(dict.fromkeys(matched_k + matched_v))))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt-file', required=True)
    ap.add_argument('--english-paths', default='main/resource/localization/abilities_english.txt.json,main/resource/localization/dota_english.txt.json,main/resource/localization/gameui_english.txt.json')
    ap.add_argument('--max-keys', type=int, default=200)
    args = ap.parse_args()

    prompt_path = Path(args.prompt_file)
    prompt = read_text(prompt_path)
    ci = detect_case_insensitive(prompt)

    english_paths = [p.strip() for p in args.english_paths.split(',') if p.strip()]
    english_paths = detect_target_files(prompt, english_paths)

    # load all maps
    en_maps: Dict[str, Dict[str, object]] = {}
    for p in english_paths:
        fp = Path(p)
        if fp.exists():
            en_maps[p] = load_json(fp)

    results: List[Dict[str, object]] = []
    collected_keys: List[str] = []

    # 1) line ranges
    ranges = parse_line_ranges(prompt)
    for hint, s, e in ranges:
        targets = english_paths
        if hint:
            # narrow
            targets = [p for p in english_paths if hint in p or hint.lower() in p.lower() or Path(p).name.lower().startswith(hint.lower()) or Path(p).name.lower() == hint.lower()]
            if not targets:
                targets = english_paths
        for p in targets:
            fp = Path(p)
            if not fp.exists():
                continue
            keys, hit_lines = extract_keys_from_line_range(fp, s, e)
            for k in keys:
                if k not in collected_keys:
                    collected_keys.append(k)
                    results.append({
                        'key': k,
                        'score': 10,  # 明示レンジは高スコア
                        'hit': 'line_range',
                        'file': p,
                        'evidence': {'line_start': s, 'line_end': e, 'lines': hit_lines},
                        'source': 'line_range',
                    })

    # 2) key contains
    if re.search(r'キー.*含む', prompt) or re.search(r'contain|contains', prompt, re.I):
        subs = quoted_tokens(prompt)
        for p, en in en_maps.items():
            keys = gather_key_contains(en, subs, ci)
            for k in keys:
                if k not in collected_keys:
                    collected_keys.append(k)
                    results.append({
                        'key': k,
                        'score': 6,
                        'hit': 'key_contains',
                        'file': p,
                        'evidence': {'substrings': subs},
                        'source': 'key_contains',
                    })

    # 3) fuzzy tokens
    qts = quoted_tokens(prompt)
    toks = camel_snake_tokens(prompt)
    tokens = list(dict.fromkeys(qts + toks))
    if tokens:
        for p, en in en_maps.items():
            hits = fuzzy_collect(en, tokens, ci)
            for k, score, hit_type, mtoks in hits:
                if k not in collected_keys:
                    collected_keys.append(k)
                    results.append({
                        'key': k,
                        'score': score,
                        'hit': hit_type,
                        'file': p,
                        'evidence': {'matched_tokens': mtoks},
                        'source': 'fuzzy_tokens',
                    })

    # sort & cap
    results.sort(key=lambda r: (-int(r['score']), r['key']))
    total_found = len(results)
    truncated = False
    if total_found > args.max_keys:
        results = results[:args.max_keys]
        truncated = True

    keys = [r['key'] for r in results]

    # outputs
    Path('keys.txt').write_text("\n".join(keys) + ("\n" if keys else ""), encoding='utf-8')
    summary = {
        'prompt_excerpt': (prompt[:400] + ('…' if len(prompt) > 400 else '')),
        'case_insensitive': ci,
        'english_paths': english_paths,
        'max_keys': args.max_keys,
        'total_found': total_found,
        'kept': len(results),
        'truncated': truncated,
        'results': results,
    }
    Path('resolved_targets.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Resolved {len(results)}/{total_found} keys. (truncated={truncated})")


if __name__ == '__main__':
    main()

