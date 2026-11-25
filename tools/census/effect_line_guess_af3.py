#!/usr/bin/env python
"""**効果線の層の推測の列 `effect_line_guess_af3`**(Track Q の頼み `note_q_census_effect_line_guess_af3_column_20261125.md`。
ユーザーの決定 11-25「推測の列として置く」)。**新しい走行ではない** —— Q が census 経由の線の列から計算した値を、sha を確かめて置く。

**推測である**(名前に guess を残す)。但し書き(列を配るときに必ず添える):
- 作者の20行(4セル × 5、寄せて引いた標本で母集団の割合ではない)で誤り 1/9(効果線と答えた行)・3/20。名前 `effect` は 3/9・7/20。**的中率ではない**
- 誤検出は背景の書き足し 2/20。描き文字・黒消しは拾わなかった
- **空 = 判じていない。「効果線でない」ではない**(訳は `status`)。vector の道具の層・.lip・アナログ原稿は範囲外

出力 `data/census_full/guess/effect_line_guess_af3_20261125.csv`(非公開。公開の census/*.csv には入れない)。
"""
import collections
import csv
import hashlib
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "inbox", "q_effect_line_guess_af3_20261125.csv")
SHA = "621eb3bad3db5704"
OUT = os.path.join(ROOT, "data", "census_full", "guess", "effect_line_guess_af3_20261125.csv")


def main() -> int:
    if not hashlib.sha256(open(SRC, "rb").read()).hexdigest().startswith(SHA):
        print(f"**sha256 が違う** {SRC}"); return 1
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    keys = {(r["page_path"], r["layer_id"]) for r in rows}
    print(f"層 {len(rows):,}(重複 {len(rows) - len(keys)})/ 推測: " + " / ".join(
        f"{k or '空(判じていない)'} {v:,}" for k, v in collections.Counter(r["effect_line_guess_af3"] for r in rows).most_common()))
    print("判じていない訳: " + " / ".join(f"{k} {v:,}" for k, v in collections.Counter(
        r["status"] for r in rows if r["effect_line_guess_af3"] == "").most_common()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    shutil.copyfile(SRC, OUT)
    print(f"出力: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
