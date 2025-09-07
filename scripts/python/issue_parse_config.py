#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Issue本文から YAML 設定ブロックを抽出して GitHub Actions の出力に書き出す。

使い方:
  python scripts/python/issue_parse_config.py <<'EOF'
  <issue body markdown>
  EOF

出力 (GITHUB_OUTPUT 環境変数に追記):
  target, english, japanese, base, head, keys(改行区切り), filter_key, filter_value

YAML例:
```yaml
target: abilities
base: origin/main
head: HEAD
keys:
  - DOTA_Tooltip_ability_foo
filter_key: "^DOTA_Tooltip_ability_"
filter_value: "Cooldown|Mana Cost"
```
"""

from __future__ import annotations
import os, re, sys, yaml


def parse_yaml_block(md: str):
    m = re.search(r"```ya?ml\s*(.*?)```", md, flags=re.S|re.I)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except Exception:
        return {}


def main():
    body = sys.stdin.read()
    data = parse_yaml_block(body)
    target = (data.get('target') or 'abilities').strip()
    mapping = {
        'abilities': (
            'main/resource/localization/abilities_english.txt.json',
            'main/resource/localization/abilities_japanese.txt.json',
        ),
        'dota': (
            'main/resource/localization/dota_english.txt.json',
            'main/resource/localization/dota_japanese.txt.json',
        ),
        'gameui': (
            'main/resource/localization/gameui_english.txt.json',
            'main/resource/localization/gameui_japanese.txt.json',
        ),
    }
    english, japanese = mapping.get(target, mapping['abilities'])

    out_lines = []
    out_lines.append(f"target={target}")
    out_lines.append(f"english={english}")
    out_lines.append(f"japanese={japanese}")
    for k in ['base', 'head', 'filter_key', 'filter_value']:
        v = data.get(k)
        if v is not None:
            out_lines.append(f"{k}={str(v)}")
    if isinstance(data.get('keys'), list) and data['keys']:
        keys = "\n".join(str(x) for x in data['keys'])
        out_lines.append(f"keys<<EOF\n{keys}\nEOF")

    gh_out = os.getenv('GITHUB_OUTPUT')
    if gh_out:
        with open(gh_out, 'a', encoding='utf-8') as f:
            for line in out_lines:
                print(line, file=f)
    else:
        print("\n".join(out_lines))


if __name__ == '__main__':
    main()

