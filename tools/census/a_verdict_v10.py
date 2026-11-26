#!/usr/bin/env python
"""**a の判定 v10 と `is_base_mask` v2** —— `下地 N` の裁定を当てる(Track E の通達 `note_census_e_a_v10_shitaji_applied_two_corrections_20261126.md`)。

**数を見る前に決めたこと**:
- 作者の裁定(11-26、逐語): 「下地Nはクリップスタジオペイントの３D素材をLT変換したときに作られるベースマスクだ 使い道がblender のレンダリングで作成されるベースと同じ」
- 集合 = 層名が**ちょうど** `下地 1`〜`下地 4` でフォルダーでない層(E の決定 (i)、161層 = .clip 160 + .lip 1)。
  `下地 1 のコピー`・`下地 2 のコピー`・`下地白 ` は**入れない**(作者が名を挙げていない。作者に聞く)
- `is_base_mask` v2 = v1(5名前 5,148層、11-23)+ 上の集合。**v1 は書き換えない**(別のファイル、列 `is_base_mask_v1` も残す)
- a v10 = v9 で上の集合の層が落としていなければ `template_excluded`、理由 `条項:ベースマスク(名前:下地N)`。v9 で既に落ちていた層は理由を書き換えない
- **札であって合成からの削除ではない**(E の 11-26 の読み方の通達)。.lip の `下地 3` 1層は土台で乗る層が 1
- E の数 219,799 / 437,425 = 0.50248 と突き合わせる。前後を名前を付けて印字する(規律10)
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
V9 = os.path.join(D, "a_verdict_v9_20261124.csv")
BM1 = os.path.join(D, "is_base_mask_20261123.csv")
OUT_A = os.path.join(D, "a_verdict_v10_20261126.csv")
OUT_B = os.path.join(D, "is_base_mask_v2_20261126.csv")
NAMES = ["下地 1", "下地 2", "下地 3", "下地 4"]
DROP = "条項:ベースマスク(名前:下地N)"
SRC = "作者の裁定 11-26: 下地N は 3D 素材を LT 変換したときに作られるベースマスク、使い道は rendering の base と同じ"
E_COUNT = 219799


def main() -> int:
    c = db.connect()
    add = {(p, str(i)) for p, i in c.execute(
        f"select page_path, layer_id from layers where name in ({','.join('?' * len(NAMES))}) and not folder", NAMES).fetchall()}
    print(f"[is_base_mask v2] 足す層 {len(add)}(" + " / ".join(f"{n} {v}" for n, v in sorted(collections.Counter(
        c.execute('select name from layers where page_path = ? and layer_id = ?', [p, int(i)]).fetchone()[0] for p, i in add).items())) + ")")
    bm, n1, n2 = [], 0, 0
    for r in csv.DictReader(open(BM1, encoding="utf-8")):
        k = (r["page_path"], r["layer_id"])
        v2 = int(r["is_base_mask"] == "1" or k in add)
        n1 += r["is_base_mask"] == "1"; n2 += v2
        bm.append(dict(page_path=k[0], layer_id=k[1], is_base_mask_v2=v2, is_base_mask_v1=r["is_base_mask"], in_a_population=r["in_a_population"]))
    print(f"    v1(11-23){n1:,} → v2(11-26){n2:,}(+{n2 - n1})/ 全層 {len(bm):,}")
    out, moved, already = [], 0, 0
    for r in csv.DictReader(open(V9, encoding="utf-8")):
        k = (r["page_path"], r["layer_id"])
        v, why, src = r["a_verdict_v9"], r["a_rule_v9"], r["a_source_v9"]
        if k in add:
            if v == "template_excluded":
                already += 1
            else:
                moved += 1; v, why, src = "template_excluded", DROP, SRC
        out.append(dict(page_path=k[0], layer_id=k[1], a_verdict_v10=v, a_rule_v10=why, a_source_v10=src,
                        a_verdict_v9=r["a_verdict_v9"], is_base_mask_v2=int(r["is_base_mask"] == "1" or k in add),
                        a_rule_v7=r["a_rule_v7"], keep_name=r["keep_name"]))
    n = len(out)
    c9 = collections.Counter(x["a_verdict_v9"] for x in out)
    c10 = collections.Counter(x["a_verdict_v10"] for x in out)
    print(f"[a の判定 v10] 動いた {moved} / 既に落ちていた {already} / a の外 {len(add) - moved - already}")
    for kk in ("template_excluded", "undecided", "not_excluded"):
        print(f"    {kk}: v9(11-24){c9[kk]:,} → v10(11-26){c10[kk]:,}")
    print(f"    落とす率: v9 {c9['template_excluded']:,} / {n:,} = {c9['template_excluded'] / n:.5f} → v10 {c10['template_excluded']:,} / {n:,} = {c10['template_excluded'] / n:.5f}")
    print(f"    E の数 {E_COUNT:,} と: " + ("一致" if c10["template_excluded"] == E_COUNT else f"**食い違い** census {c10['template_excluded']:,}"))
    print(f"    ベースマスク v2 の層で落とさないもの: {sum(1 for x in out if x['is_base_mask_v2'] == 1 and x['a_verdict_v10'] != 'template_excluded')}")
    for f, rows in ((OUT_B, bm), (OUT_A, out)):
        with open(f, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"    → {f}({len(rows):,}行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
