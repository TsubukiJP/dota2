import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Any, Tuple
import subprocess
import time
import random
import csv

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 定数
MODEL_NAME = 'gemini-2.5-flash'
BATCH_SIZE = 20

# GitHub Actions 用ログヘルパー
def log_separator(title: str = None):
    """区切り線"""
    print("="*50, flush=True)
    if title:
        print(f"[{title}]", flush=True)
        print("="*50, flush=True)

def log_warning_group(key: str, errors: List[str], context: Dict[str, Any] = None):
    """検証エラーを構造化された形式で出力します。"""
    from collections import Counter
    
    # エラー種別を判定
    kind = "validation_error"
    kind_label = "検証エラー"
    rule = "unknown"
    err_text = "\n".join(errors) if errors else ""
    if "プレースホルダー" in err_text or "タグ" in err_text:
        kind = "token_mismatch"
        kind_label = "トークン不一致"
        rule = "placeholders_must_match"
    elif "KEEP" in err_text:
        kind = "keep_term_missing"
        kind_label = "KEEP用語欠落"
        rule = "keep_terms_required"
    elif "FORCE" in err_text:
        kind = "force_term_missing"
        kind_label = "FORCE用語欠落"
        rule = "force_terms_preferred"
    
    # 親ログ: 動的なkind表示
    print(f"::warning::{kind}({kind_label}) key={key}", flush=True)
    print(f"::group::detail key={key}", flush=True)
    print(f"  kind: {kind}", flush=True)
    print(f"  rule: {rule}", flush=True)
    print("", flush=True)
    
    if context:
        orig_tokens = context.get('orig_tokens') or []
        tran_tokens = context.get('tran_tokens') or []
        
        # トークン表示（カウント付き）
        print(f"  tokens:", flush=True)
        print(f"    orig: {orig_tokens} ({len(orig_tokens)})", flush=True)
        print(f"    tran: {tran_tokens} ({len(tran_tokens)})", flush=True)
        print("", flush=True)
        
        # diff: Counter で多重集合の差分を計算
        oc = Counter(orig_tokens)
        tc = Counter(tran_tokens)
        missing = list((oc - tc).elements())
        extra = list((tc - oc).elements())
        
        if missing or extra:
            print(f"  diff:", flush=True)
            if missing:
                print(f"    missing: {missing}", flush=True)
            if extra:
                print(f"    extra: {extra}", flush=True)
            print("", flush=True)
        
        # テキスト表示（json.dumpsでエスケープ）
        orig_text = context.get('orig_text') or ''
        tran_text = context.get('tran_text') or ''
        if orig_text or tran_text:
            orig_preview = orig_text[:80] + "..." if len(orig_text) > 80 else orig_text
            tran_preview = tran_text[:80] + "..." if len(tran_text) > 80 else tran_text
            print(f"  text:", flush=True)
            print(f"    orig: {json.dumps(orig_preview, ensure_ascii=False)}", flush=True)
            print(f"    tran: {json.dumps(tran_preview, ensure_ascii=False)}", flush=True)
            print("", flush=True)
    
    # エラー詳細
    if errors:
        print("", flush=True)
        print(f"  errors:", flush=True)
        for e in errors[:3]:
            print(f"    - {e}", flush=True)
    
    print("::endgroup::", flush=True)




def log_progress(current: int, total: int, filename: str, count: int):
    """進捗"""
    print(f"[{current}/{total}] 翻訳中: {filename} ({count}件)", flush=True)



def load_profile_config(docs_dir: str) -> Dict[str, Any]:
    """プロファイル設定ファイル (YAML) を読み込みます。"""
    import yaml
    config_path = os.path.join(docs_dir, "profile_config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"プロファイル設定ファイルが見つかりません: {config_path}")
        return {}
    except Exception as e:
        logger.error(f"プロファイル設定ファイルのパースエラー: {e}")
        return {}

def get_profile_for_file(filepath: str, config: Dict[str, Any] = None) -> str:
    """ファイルパスに基づいて適用するプロファイルを決定します。"""
    basename = os.path.basename(filepath).lower().replace('.txt.json', '').replace('.json', '')
    
    if config and 'profiles' in config:
        for profile_name, profile_info in config['profiles'].items():
            for pattern in profile_info.get('pattern', []):
                if pattern.lower() in basename:
                    return profile_name
    
    # フォールバック
    if 'abilities_english' in basename:
        return 'abilities'
    if 'addon_english' in basename:
        return 'event'
    if 'gameui' in basename or 'richpresence' in basename:
        return 'ui_simple'
    return 'ui_complex'

def load_rules_for_profile(docs_dir: str, profile: str, config: Dict[str, Any] = None) -> str:
    """プロファイルに応じたルールファイルを結合して読み込みます。
    
    結合順序: common.prepend → profile.rules → common.append
    """
    rule_files = []
    
    # common.prepend を先頭に追加
    if config and 'common' in config:
        prepend = config['common'].get('prepend')
        if prepend:
            rule_files.append(prepend)
    
    # プロファイル固有のルールを追加
    if config and 'profiles' in config and profile in config['profiles']:
        profile_rules = config['profiles'][profile].get('rules', [])
        rule_files.extend(profile_rules)
    else:
        logger.warning(f"プロファイル '{profile}' の設定が見つかりません。")
    
    # common.append を最後に追加
    if config and 'common' in config:
        append = config['common'].get('append')
        if append:
            rule_files.append(append)
    
    # フォールバック
    if not rule_files:
        rule_files = ['ai_rules_base.md', 'ai_rules_checklist.md']
        logger.warning("デフォルトルールを使用します。")
    
    # ルールファイルを結合
    rules_content = ""
    for rule_file in rule_files:
        rule_path = os.path.join(docs_dir, rule_file)
        if os.path.exists(rule_path):
            try:
                with open(rule_path, 'r', encoding='utf-8') as f:
                    if rules_content:
                        rules_content += "\n\n---\n\n" + f.read()
                    else:
                        rules_content = f.read()
                logger.info(f"ルールファイルを読み込みました: {rule_file}")
            except Exception as e:
                logger.warning(f"ルールファイルの読み込みエラー ({rule_file}): {e}")
        else:
            logger.warning(f"ルールファイルが見つかりません: {rule_path}")
    
    return rules_content

def load_json_file(filepath: str) -> Dict[str, Any]:
    """JSONファイルを読み込み、その内容を返します。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"ファイルが見つかりません: {filepath}。空の辞書を返します。")
        return {}
    except json.JSONDecodeError:
        logger.error(f"JSONのデコードに失敗しました: {filepath}")
        sys.exit(1)

def save_json_file(filepath: str, data: Dict[str, Any]):
    """データをJSONファイルに保存します。"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"保存しました: {filepath}")
    except Exception as e:
        logger.error(f"ファイルの保存に失敗しました {filepath}: {e}")
        sys.exit(1)

def safe_json_loads(s: str) -> Dict[str, Any]:
    """JSONデコードを試み、失敗時はマークダウンブロック等の除去を試みて再パースします。"""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # 最初の { から最後の } までを抽出
        l = s.find("{")
        r = s.rfind("}")
        if l != -1 and r != -1 and l < r:
            try:
                return json.loads(s[l:r+1])
            except json.JSONDecodeError:
                pass # それでもダメなら元のエラーを出すためにraiseしない
    
    # 失敗時は例外を再送出するが、呼び出し元でキャッチしてもらうため
    # ここでは元の文字列を含めたエラーとしてログに出すか、Noneを返す設計も手だが
    # 今回は translate_batch 内で try-except しているので raise で良い
    raise json.JSONDecodeError("Failed to parse JSON even with cleanup", s, 0)

def load_glossary(filepath: str) -> List[Dict[str, str]]:
    """CSVファイルから用語集を読み込み、英語の用語の長さ（降順）でソートします。"""
    glossary = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['原文'] and row['訳語']:
                    glossary.append(row)
    except FileNotFoundError:
        logger.error(f"用語集ファイルが見つかりません: {filepath}")
        sys.exit(1)
    
    # 最長一致のために長さを降順ソート
    glossary.sort(key=lambda x: len(x['原文']), reverse=True)
    return glossary

def extract_relevant_glossary_terms(text: str, glossary: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    テキストに含まれる用語を用語集から抽出します。
    最長一致で占有し、重複マッチを防ぎます。
    Word Boundary (\b) の代わりに (?<![A-Za-z0-9]) を使用します。
    """
    relevant_terms = []
    
    # 処理用のテキスト（マッチした部分をマスクしていく）
    # 大文字小文字を区別しない検索を行うため、検索時は lower を使うが、置換位置はずれないように注意
    # ただし regex の IGNORECASE を使うので text 自体は保持する
    # マスクリング戦略: マッチした箇所を特殊な文字で埋める
    
    working_text = text
    
    for entry in glossary:
        term = entry['原文']
        # 堅牢な境界チェック
        pattern_str = r'(?<![A-Za-z0-9])' + re.escape(term) + r'(?![A-Za-z0-9])'
        
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            
            # 検索
            if pattern.search(working_text):
                relevant_terms.append(entry)
                # マッチした部分をマスクして、より短い用語が部分一致しないようにする
                # 例: 'Cast Range' を見つけたら '__________' に変換して 'Range' がヒットしないようにする
                working_text = pattern.sub(lambda m: '_' * len(m.group(0)), working_text)
                
        except re.error:
            logger.warning(f"無効な正規表現パターンです: {term}")

    return relevant_terms

def get_keys_from_json(data: Dict[str, Any]) -> Dict[str, str]:
    """JSONをフラット化してキーと値のペアを取得します ('Tokens' 構造を処理)。"""
    tokens = {}
    if not data: return {}
    
    def extract_recursive(d):
        for k, v in d.items():
            if k == "Tokens" and isinstance(v, dict):
                 tokens.update(v)
            elif isinstance(v, dict):
                 extract_recursive(v)
            elif isinstance(v, str):
                 pass 
    
    extract_recursive(data)
    
    # トークンが見つからない場合のフォールバック
    if not tokens:
        if any(isinstance(v, str) for v in data.values()):
             return data
    
    return tokens

def get_old_file_content(filepath: str) -> Dict[str, Any]:
    """前のコミット (HEAD~1) からファイルの内容を取得します。"""
    try:
        rel_path = os.path.relpath(filepath, os.getcwd())
        rel_path = rel_path.replace(os.sep, '/')
        
        result = subprocess.run(
            ['git', 'show', f'HEAD~1:{rel_path}'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError:
        logger.warning(f"旧コンテンツの取得に失敗しました (新規ファイルの可能性があります): {filepath}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"旧JSONのデコードに失敗しました: {filepath}")
        return {}
    except Exception as e:
        logger.error(f"旧ファイルコンテンツの取得エラー: {e}")
        return {}

def get_keys_in_line_range(filepath: str, from_line: int = 1, to_line: int = None) -> set:
    """指定された行範囲内にあるJSONキーを抽出します。
    
    Args:
        filepath: JSONファイルのパス
        from_line: 開始行番号 (1-indexed)
        to_line: 終了行番号 (1-indexed、Noneの場合はファイル末尾まで)
    
    Returns:
        set: 行範囲内に存在するキーのセット
    """
    keys_in_range = set()
    key_pattern = re.compile(r'^\s*"([^"]+)"\s*:\s*"')
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if to_line is None:
            to_line = len(lines)
        
        # 1-indexed -> 0-indexed
        start_idx = max(0, from_line - 1)
        end_idx = min(len(lines), to_line)
        
        for i in range(start_idx, end_idx):
            line = lines[i]
            match = key_pattern.match(line)
            if match:
                keys_in_range.add(match.group(1))
        
        logger.info(f"行範囲 {from_line}-{to_line} から {len(keys_in_range)} 件のキーを検出")
        return keys_in_range
        
    except Exception as e:
        logger.error(f"行範囲キー取得エラー: {e}")
        return set()

def get_diff_keys(base_commit: str = "HEAD~1") -> Dict[str, set]:
    """指定されたコミットとの差分から追加/変更されたキーを抽出します。
    
    Returns:
        Dict[str, set]: ファイル名をキー、変更されたキーのセットを値とする辞書
    """
    diff_keys = {}  # {filename: {key1, key2, ...}}
    
    try:
        # git diff で変更行を取得
        result = subprocess.run(
            ["git", "diff", base_commit, "HEAD", "--unified=0", "--", "*_english.txt.json"],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode != 0:
            logger.warning(f"git diff 実行エラー: {result.stderr}")
            return {}
        
        current_file = None
        key_pattern = re.compile(r'^\+\s*"([^"]+)"\s*:\s*"')
        file_pattern = re.compile(r'^\+\+\+ b/(.+)$')
        
        for line in result.stdout.split('\n'):
            # ファイル名を検出
            file_match = file_pattern.match(line)
            if file_match:
                current_file = os.path.basename(file_match.group(1))
                if current_file not in diff_keys:
                    diff_keys[current_file] = set()
                continue
            
            # 追加された行からキーを抽出 (+で始まる行、ただし+++は除く)
            if line.startswith('+') and not line.startswith('+++'):
                key_match = key_pattern.match(line)
                if key_match and current_file:
                    diff_keys[current_file].add(key_match.group(1))
        
        total_keys = sum(len(keys) for keys in diff_keys.values())
        logger.info(f"差分から {total_keys} 件のキーを検出 ({len(diff_keys)} ファイル)")
        
        return diff_keys
        
    except Exception as e:
        logger.error(f"差分キーの取得エラー: {e}")
        return {}


def identify_changes(english_data: Dict[str, Any], japanese_data: Dict[str, Any], filepath: str, mode: str, diff_keys: set = None, force: bool = False) -> Dict[str, str]:
    """翻訳が必要なキーを特定します。"""
    to_translate = {}
    
    eng_tokens = get_keys_from_json(english_data)
    jp_tokens = get_keys_from_json(japanese_data)

    for key, eng_text in eng_tokens.items():
        if not isinstance(eng_text, str): continue
        if not eng_text.strip(): continue 
        
        needs_translation = False
        jp_text = jp_tokens.get(key)
        
        # diffモード: 差分キーのみを対象
        if mode == 'diff':
            if diff_keys and key in diff_keys:
                needs_translation = True
        
        # targetモード: 未翻訳のみ（forceなら全て）
        elif mode == 'target':
            if force:
                needs_translation = True
            elif not jp_text:
                needs_translation = True
        
        if needs_translation:
            to_translate[key] = eng_text
            
    return to_translate

# トークンパターン
TOKEN_PATTERN = re.compile(r"""
(
    %%                            | # リテラル%（100%%など）
    %[sdif]\d+                    | # %s1, %d2 など（番号付きprintf）
    %[sdif](?![a-zA-Z0-9_])       | # %d, %s, %f など（単体printf、後ろに英数字が続かない）
    %[A-Za-z_][A-Za-z0-9_]*%      | # %variable_name%（英字/_開始の変数）
    %\$(?:str|agi|int)\b          | # %$str など（ステータスボーナス）
    \$(?:str|agi|int)\b           | # $str など（%なしバージョン）
    \{[gsd]:[^}]+\}               | # {g:xxx} {s:xxx} {d:xxx}（テンプレート変数）
    <[^>]+>                       | # HTMLタグ
    \\n                            # エスケープ改行
)
""", re.VERBOSE)



def debug_token_diff(original: str, translated: str):
    """トークン抽出結果の差異をデバッグログに出力します。"""
    o = TOKEN_PATTERN.findall(original)
    t = TOKEN_PATTERN.findall(translated)
    if o != t:
        logger.warning(f"TOKEN DIFF\nORIG: {o}\nTRAN: {t}")

def validate_translation(original: str, translated: str, term_check_list: List[Dict[str, str]], strict_force: bool = False) -> Tuple[bool, List[str], Dict[str, Any]]:
    """ルールに基づいて翻訳を検証します。
    
    Returns:
        Tuple[bool, List[str], Dict[str, Any]]: (有効か, エラーリスト, トークン情報)
    """
    errors = []
    token_info = {}
    
    # 1. プレースホルダーとタグ (FAIL レベル - 常に厳格)
    orig_matches = TOKEN_PATTERN.findall(original)
    trans_matches = TOKEN_PATTERN.findall(translated)
    
    token_info['orig_tokens'] = orig_matches
    token_info['tran_tokens'] = trans_matches
    token_info['orig_text'] = original
    token_info['tran_text'] = translated
    
    # 順序も含めて完全一致する必要がある
    if orig_matches != trans_matches:
        debug_token_diff(original, translated)
        errors.append(f"プレースホルダー/タグの不一致: 原文 {orig_matches} vs 訳文 {trans_matches}")
    
    # 2. KEEP 用語 (FAIL レベル - 常に厳格)
    for entry in term_check_list:
        if entry['制約'] == 'KEEP':
            term = entry['原文']
            if term not in translated:
                 errors.append(f"KEEP用語の欠落: {term}")

    # 3. FORCE 用語 (デフォルト WARN / --strict-force で FAIL)
    for entry in term_check_list:
        if entry['制約'] == 'FORCE':
            target = entry['訳語']
            if target not in translated:
                msg = f"FORCE翻訳の欠落の可能性: {entry['原文']} -> 期待値 {target}"
                if strict_force:
                    errors.append(msg)
                else:
                    # 警告のみ
                    logger.warning(msg)
    
    return len(errors) == 0, errors, token_info

def translate_batch(batch: Dict[str, str], glossary: List[Dict[str, str]], ai_rules: str, strict_force: bool = False) -> Dict[str, str]:
    """Gemini APIを使用してバッチ翻訳を実行します。"""
    if not batch: return {}
    
    # 1. このバッチに関連する用語を抽出
    batch_text = " ".join(batch.values())
    relevant_terms = extract_relevant_glossary_terms(batch_text, glossary)
    glossary_text = "\n".join([f"{item['原文']} -> {item['訳語']} ({item['制約']})" for item in relevant_terms])
    
    # 2. プロンプト作成
    system_instruction = f"""You are a professional translator for Dota 2.
    
{ai_rules}

## Glossary (Strict priority)
{glossary_text}
"""
    
    prompt = f"""Translate the following English Dota 2 texts to Japanese.
Output strictly in JSON format: {{ "key": "translated_text" }}.

Input:
{json.dumps(batch, ensure_ascii=False, indent=2)}
"""

    max_retries = 3
    base_delay = 2
    
    # モデル設定はmainで実施済みと想定するが、インスタンス化はここで行う
    model = genai.GenerativeModel(MODEL_NAME, system_instruction=system_instruction)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            cleaned_text = response.text.strip()
            # safe_json_loads でパース
            translated_data = safe_json_loads(cleaned_text)
            
            # 検証
            validated_data = {}
            for key, translated_text in translated_data.items():
                original_text = batch.get(key, "")
                key_terms = extract_relevant_glossary_terms(original_text, glossary)
                is_valid, errors, token_info = validate_translation(original_text, translated_text, key_terms, strict_force)
                
                if is_valid:
                    validated_data[key] = translated_text
                else:
                    context = {
                        'orig_tokens': token_info.get('orig_tokens', []),
                        'tran_tokens': token_info.get('tran_tokens', []),
                        'orig_text': token_info.get('orig_text', ''),
                        'tran_text': token_info.get('tran_text', '')
                    }
                    log_warning_group(key, errors, context)
            
            return validated_data

        except Exception as e:
            logger.error(f"API呼び出し失敗 (試行 {attempt+1}/{max_retries}): {e}")
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))
    
    return {} # リトライ後も失敗

def parse_arguments():
    """コマンドライン引数を解析します。"""
    parser = argparse.ArgumentParser(description='Gemini APIを使用したDota 2自動翻訳スクリプト')
    
    parser.add_argument('--mode', choices=['diff', 'target'], default='diff',
                        help='翻訳モード: diff (コミット差分), target (ファイル・範囲指定)。')
    parser.add_argument('--base-commit', type=str, default='HEAD~1',
                        help='diffモードで比較するベースコミット (デフォルト: HEAD~1)。')
    parser.add_argument('--max-items', type=int, default=0,
                        help='翻訳する最大キー数（0は無制限）。')
    parser.add_argument('--priority', type=str, nargs='+',
                        help='優先的に翻訳するカテゴリ（例: Lore Description）。キー名に含まれる文字列で判定します。')
    parser.add_argument('--strict-unprocessed', action='store_true',
                        help='未処理キーがある場合に終了コード1で終了します。')
    parser.add_argument('--strict-force', action='store_true',
                        help='FORCE用語の制約違反（警告レベル）をエラーとして扱います。')
    parser.add_argument('--force', action='store_true',
                        help='targetモードで既存の翻訳も含めて再翻訳します。')
    
    # ターゲットファイル・行範囲指定
    parser.add_argument('--target-file', type=str, default=None,
                        help='翻訳対象のファイル名 (例: dota_english.txt.json)。targetモードでは必須。')
    parser.add_argument('--from-line', type=int, default=1,
                        help='--target-file指定時の開始行番号 (デフォルト: 1)。')
    parser.add_argument('--to-line', type=int, default=None,
                        help='--target-file指定時の終了行番号 (デフォルト: ファイル末尾)。')
    
    return parser.parse_args()

def main():
    start_time = time.time()
    args = parse_arguments()
    
    # target モードのバリデーション
    if args.mode == 'target' and not args.target_file:
        logger.error("エラー: --mode target には --target-file が必須です。")
        sys.exit(1)
    
    # 開始バナー
    log_separator("START")
    print(f"翻訳プロセス開始", flush=True)
    print(f"モード: {args.mode}", flush=True)
    if args.mode == 'diff':
        print(f"基準コミット: {args.base_commit}", flush=True)
    if args.mode == 'target':
        print(f"対象ファイル: {args.target_file} (行: {args.from_line}-{args.to_line or 'EOF'})", flush=True)
        if args.force:
            print(f"強制再翻訳: 有効", flush=True)
    log_separator()
    
    # APIキーチェック
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        logger.error("重大エラー: GEMINI_API_KEY が設定されていません。終了します。")
        sys.exit(1)
    
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    
    base_dir = "." 
    resource_dir = os.path.join(base_dir, "main", "resource", "localization")
    docs_dir = os.path.join(base_dir, "docs")
    
    glossary_path = os.path.join(docs_dir, "glossary.csv")
        
    glossary = load_glossary(glossary_path)
    
    # プロファイル設定を読み込む
    profile_config = load_profile_config(docs_dir)
    
    target_files = []
    if os.path.exists(resource_dir):
        for f in os.listdir(resource_dir):
            if f.endswith("_english.txt.json"):
                target_files.append(os.path.join(resource_dir, f))
    else:
        logger.error(f"リソースディレクトリが見つかりません: {resource_dir}")
        sys.exit(1)
    
    # --target-file が指定されている場合、そのファイルのみを処理
    if args.target_file:
        target_files = [f for f in target_files if args.target_file in f]
        if not target_files:
            logger.error(f"指定されたファイルが見つかりません: {args.target_file}")
            sys.exit(1)
        logger.info(f"対象ファイル: {target_files[0]} (行範囲: {args.from_line}-{args.to_line or 'EOF'})")
    
    # 行範囲フィルタ用のキーセット
    line_range_keys = None
    if args.target_file and (args.from_line > 1 or args.to_line is not None):
        line_range_keys = get_keys_in_line_range(target_files[0], args.from_line, args.to_line)
        
    total_processed_count = 0
    total_batch_count = 0
    total_violation_count = 0
    files_processed = 0
    translated_files_list = []
    unprocessed_keys_report = {} # Key: {file_path, error}

    for file_path in target_files:
        jp_file_path = file_path.replace("_english.txt.json", "_japanese.txt.json")
        
        # ファイルに応じたプロファイルを選択してルールを読み込む
        profile = get_profile_for_file(file_path, profile_config)
        ai_rules = load_rules_for_profile(docs_dir, profile, profile_config)
        logger.info(f"処理中: {file_path} (プロファイル: {profile})")

        
        eng_data = load_json_file(file_path)
        jp_data = load_json_file(jp_file_path)
        
        # diffモードの場合は事前に差分キーを取得
        file_diff_keys = None
        if args.mode == 'diff':
            if 'all_diff_keys' not in dir():
                all_diff_keys = get_diff_keys(args.base_commit)
            basename = os.path.basename(file_path)
            file_diff_keys = all_diff_keys.get(basename, set())
            if not file_diff_keys:
                logger.info(f"差分キーがありません: {basename}")
                continue
        
        to_translate_map = identify_changes(eng_data, jp_data, file_path, args.mode, file_diff_keys, args.force)
        
        # 行範囲フィルタが有効な場合、キーをフィルタリング
        if line_range_keys is not None:
            filtered_map = {k: v for k, v in to_translate_map.items() if k in line_range_keys}
            logger.info(f"行範囲フィルタ適用: {len(to_translate_map)} -> {len(filtered_map)} 件")
            to_translate_map = filtered_map
        
        keys_to_process = list(to_translate_map.keys())
        
        # 優先度ソートの実装
        if args.priority:
            def priority_score(key):
                for i, p in enumerate(args.priority):
                    if p in key:
                        return len(args.priority) - i 
                return -1
            
            keys_to_process.sort(key=priority_score, reverse=True)

        if args.max_items > 0 and total_processed_count + len(keys_to_process) > args.max_items:
             remaining_slots = args.max_items - total_processed_count
             if remaining_slots <= 0:
                 logger.info("最大アイテム数に達しました。停止します。")
                 break
             keys_to_process = keys_to_process[:remaining_slots]
        
        if not keys_to_process:
            logger.info("このファイルには翻訳対象がありません。")
            continue
            
        total_processed_count += len(keys_to_process)
        
        batch_size = 20
        jp_tokens_ref = None
        
        # JPデータの書き込み先特定
        stack = [jp_data]
        while stack:
            curr = stack.pop()
            if isinstance(curr, dict):
                if "Tokens" in curr and isinstance(curr["Tokens"], dict):
                    jp_tokens_ref = curr["Tokens"]
                    break
                for v in curr.values():
                    if isinstance(v, dict):
                        stack.append(v)
        
        if jp_tokens_ref is None:
            if not jp_data:
                jp_data = {"lang": {"Language": "Japanese", "Tokens": {}}}
                jp_tokens_ref = jp_data["lang"]["Tokens"]
            else:
                 jp_tokens_ref = jp_data
        
        # バッチ処理
        for i in range(0, len(keys_to_process), batch_size):
            batch_keys = keys_to_process[i:i+batch_size]
            batch_dict = {k: to_translate_map[k] for k in batch_keys}
            
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(keys_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
            log_progress(batch_num, total_batches, os.path.basename(file_path), len(batch_dict))
            total_batch_count += 1
            
            # ここで --strict-force を渡す
            translated_batch = translate_batch(batch_dict, glossary, ai_rules, args.strict_force)
            
            for k, text in translated_batch.items():
                jp_tokens_ref[k] = text
            
            for k in batch_dict:
                if k not in translated_batch:
                     unprocessed_keys_report[k] = {"file": os.path.basename(file_path), "error": "検証失敗またはAPIエラー"}
                     total_violation_count += 1

        files_processed += 1
        translated_files_list.append(os.path.basename(file_path))

        save_json_file(jp_file_path, jp_data)

    if unprocessed_keys_report:
        logger.warning(f"未処理キー数: {len(unprocessed_keys_report)}")
        preview = list(unprocessed_keys_report.items())[:5]
        logger.warning(f"プレビュー: {preview}")
        
        # 未処理キーをファイルに出力
        report_file = "unprocessed_keys.log"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(unprocessed_keys_report, f, indent=4, ensure_ascii=False)
            logger.info(f"未処理キーレポートを保存しました: {report_file}")
        except Exception as e:
            logger.error(f"レポート保存失敗: {e}")

        # --strict-unprocessed のチェック
        if args.strict_unprocessed:
            logger.error("Strict(Unprocessed)モード有効: 未処理キーが存在するため異常終了します。")
            sys.exit(1)

    # 完了サマリー
    elapsed_time = time.time() - start_time
    elapsed_min = int(elapsed_time // 60)
    elapsed_sec = int(elapsed_time % 60)
    
    log_separator("SUMMARY")
    print(f"翻訳完了", flush=True)
    print(f"処理ファイル: {files_processed}", flush=True)
    print(f"バッチ数: {total_batch_count}", flush=True)
    print(f"翻訳キー数: {total_processed_count}", flush=True)
    print(f"違反数: {total_violation_count}", flush=True)
    print(f"所要時間: {elapsed_min}分{elapsed_sec}秒", flush=True)
    log_separator()
    
    # サマリーJSONを出力（ワークフローで使用）
    summary_data = {
        "model": MODEL_NAME,
        "mode": args.mode,
        "files_processed": files_processed,
        "batch_count": total_batch_count,
        "translation_count": total_processed_count,
        "violation_count": total_violation_count,
        "elapsed_time": f"{elapsed_min}分{elapsed_sec}秒",
        "translated_files": translated_files_list
    }
    
    try:
        with open("translation_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        logger.info("サマリーを保存しました: translation_summary.json")
    except Exception as e:
        logger.error(f"サマリー保存失敗: {e}")

if __name__ == "__main__":
    main()
