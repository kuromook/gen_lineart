#!/usr/bin/env python
"""**描き文字の層の推測の列 `kakimoji_guess`**(Track Q の頼み `note_q_census_kakimoji_guess_column_20261125.md`。
ユーザーの決定 11-25「推測の列として置く」)。**新しい走行ではない** —— Q が census の Q30 の一覧と効果線の推測の列から計算した値を、sha を確かめて置く。

**推測である**(名前に guess を残す)。規則: N = 名前に 描き文字・書き文字・かき文字 を含み `下絵` で始まらない(フォルダー・テキスト・空・.lip・形式5 は判じない)/
E = 名前に `effect` を含み、効果線の推測が 0 → 1、1 → 0。但し書き(列を配るときに必ず添える):
- 作者の20行(群に寄せて引いた標本で割合ではない): E の 1 で 描き文字 8/10(外れ: 線画 1・黒消し 1)/ E の 0 で 効果線 5/5 / N で 描き文字 5/5
- **名前 effect には黒消しが混ざる**。E の 1 を「黒消しでない」と読まない
- **空 = 判じていない。「描き文字でない」ではない**(訳は `status`)。名前 effect で効果線の推測が空の層・名前の無い描き文字・線画に直接描いた描き文字は範囲の外
- 1層に描き文字と書き足しが混ざる層がある(一言 2/20)

出力 `data/census_full/guess/kakimoji_guess_20261125.csv`(非公開。公開の census/*.csv には入れない)。
"""
import collections
import csv
import hashlib
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "inbox", "q_kakimoji_guess_20261125.csv")
SHA = "e55750e8a694018a"
OUT = os.path.join(ROOT, "data", "census_full", "guess", "kakimoji_guess_20261125.csv")


def main() -> int:
    if not hashlib.sha256(open(SRC, "rb").read()).hexdigest().startswith(SHA):
        print(f"**sha256 が違う** {SRC}"); return 1
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    keys = {(r["page_path"], r["layer_id"]) for r in rows}
    print(f"層 {len(rows):,}(重複 {len(rows) - len(keys)})/ 規則 × 推測: " + " / ".join(
        f"{k[0][:1]} {k[1] or '空'} {v:,}" for k, v in sorted(collections.Counter((r["rule"], r["kakimoji_guess"]) for r in rows).items())))
    print("判じていない訳: " + " / ".join(f"{k} {v:,}" for k, v in collections.Counter(
        r["status"] for r in rows if r["kakimoji_guess"] == "").most_common()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    shutil.copyfile(SRC, OUT)
    print(f"出力: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
