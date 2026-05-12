"""
讀取 docs/corpus 下所有 .txt，呼叫 LLM 抽取事實，寫入 docs/kg_triples.txt。
僅保留「與 Lab 2 ingest 相同文法」且可通過 triples_parse.parse() 的句子。
"""
import argparse
import os
import re
from pathlib import Path

from langchain_openai import ChatOpenAI

from triples_parse import parse

LAB5 = Path(__file__).resolve().parent
CORPUS_DIR = LAB5 / "docs" / "corpus"
DEFAULT_OUT = LAB5 / "docs" / "kg_triples.txt"

os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:18299/v1"
LLM_MODEL = "Qwen2.5-3B-Instruct"

# TODO 1: 撰寫 EXTRACTION_PROMPT — 讓 LLM 從語料中抽取三元組
# 這個 prompt 需要：
#   - 說明角色（知識圖譜抽取助手）
#   - 列出五種合法句式：works_at, produces, partners_with, supplies...to, leads
#   - 設定規則：只輸出純英文句子、不推測、不加編號說明
#   - 在適當位置放入 __CORPUS__（稍後會被語料內容取代）
#   - 提供中文關鍵詞到英文關係的對應提示（例如「策略聯盟」→ partners_with）
# 可參考 docs/kg_triples.template.txt 了解五種句式的格式
EXTRACTION_PROMPT = """
你是一位專業的知識圖譜工程師。請從語料中抽取實體關係，並嚴格遵守以下格式輸出。

### 格式規範 (必須完全一致，結尾必須有句點)：
1. EntityA works_at CompanyB.
2. CompanyA produces ProductB.
3. CompanyA partners_with CompanyB.
4. CompanyA supplies ProductB to CompanyC.
5. PersonA leads ProductB.

### 規則：
- 關係名必須是小寫（works_at, produces, partners_with, supplies...to, leads）。
- 實體名稱保留原始大小寫。
- 每行一個三元組，結尾「必須」加上英文句點「.」。
- 不要加任何括號 ()、箭頭 -> 或冒號 :。

[語料內容]
__CORPUS__

### 抽取結果：
"""


def load_corpus() -> str:
    if not CORPUS_DIR.is_dir():
        raise FileNotFoundError(f"找不到語料目錄：{CORPUS_DIR}")
    parts = []
    for p in sorted(CORPUS_DIR.glob("**/*.txt")):
        body = p.read_text(encoding="utf-8").strip()
        if body:
            parts.append(f"=== 檔案: {p.name} ===\n{body}")
    if not parts:
        raise RuntimeError(f"{CORPUS_DIR} 內沒有任何 .txt")
    return "\n\n".join(parts)


def normalize_line(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^[\-\*]\s+", "", s)
    s = re.sub(r"^\d+[\.\)、]\s*", "", s)
    s = s.strip("`").strip()
    return s


def extract_raw_lines(llm_text: str) -> list[str]:
    text = llm_text.strip()
    if "```" in text:
        m = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    out = []
    for line in text.splitlines():
        n = normalize_line(line)
        if n and not n.startswith("#"):
            out.append(n)
    return out


def filter_parsable(lines: list[str]) -> tuple[list[str], list[str]]:
    good, bad = [], []
    seen = set()

    for line in lines:
        clean_line = line.strip() # 去掉換行與前後空格即可
        
        if parse(clean_line) is not None:
            if clean_line not in seen:
                good.append(clean_line)
                seen.add(clean_line)
        else:
            bad.append(line)
            
    return good, bad


def main() -> None:
    ap = argparse.ArgumentParser(description="從 corpus 經 LLM 產生 kg_triples.txt")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只印出結果，不寫入檔案",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"輸出路徑（預設：{DEFAULT_OUT}）",
    )
    args = ap.parse_args()

    corpus = load_corpus()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    prompt = EXTRACTION_PROMPT.replace("__CORPUS__", corpus)
    resp = llm.invoke(prompt).content
    raw = extract_raw_lines(resp)
    good, bad = filter_parsable(raw)

    if bad:
        print("以下行無法通過 ingest 文法，已捨棄：")
        for b in bad[:20]:
            print("  ", repr(b)[:120])
        if len(bad) > 20:
            print(f"  ... 另有 {len(bad) - 20} 行")

    header = (
        "# 本檔由 extract_triples_from_corpus.py 產生；可人工增刪後再執行 ingest_graph.py\n"
        f"# 模型：{LLM_MODEL}，temperature=0\n"
        "\n"
    )
    body = "\n".join(good) + ("\n" if good else "")

    if args.dry_run:
        print(header)
        print(body)
        print(f"# （共 {len(good)} 行可匯入）")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + body, encoding="utf-8")
    print(f"已寫入 {args.output} ，可匯入行數：{len(good)}")


if __name__ == "__main__":
    main()
