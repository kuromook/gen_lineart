#!/usr/bin/env python
"""**Q12 の母集団の頁ごとの、漫画かイラストかの表の値**(Track Q の問い、2026-11-22。既存の表の引き直し。新しい列ではない)。

- 頁 = コマ枠フォルダーの無い 4,531頁(`q_frame_likeness_order_20261120.csv` の `main`)のうち
  (a) 清書の層で画素のある層が在る(`q_finished_pages_name_rule_20261122.csv` の `n_清書_with_px` > 0)
  または (b) `cand_cls` = `名前の候補(コマN・枠型)` の層が在る —— Q の和集合
- 表 = 漫画かイラストかの表 `work_folder_manga_or_illust_20261120.csv`(作者の答え 59・未確認 212 フォルダー)
- **当てる単位 = 作品フォルダー**。頁の `work_dir`(枠らしさの一覧の列)と表の `work_dir` の完全一致。
  一致しなければ「頁の路が表のフォルダーの下に在る」でも探す(入れ子の取り違えの検出。表の中に入れ子は無い)
- 表に無い頁は、色の区分の頁の表 `colour_class_20261107.csv` で理由を引く
  (聞き取りの表は**カラーの層のある頁を1頁以上持つフォルダーだけ**を並べた。全頁がモノクロに決まったフォルダーは聞いていない)
- 落としたもの・当たらなかったものは数えて印字する(規律9)

出力 `data/census_full/q12_pages_manga_illust_20261122.csv`(非公開)。
"""
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
ORDER = os.path.join(ROOT, "outbox", "q_frame_likeness_order_20261120.csv")
FIN = os.path.join(D, "q_finished_pages_name_rule_20261122.csv")
TABLE = os.path.join(D, "work_folder_manga_or_illust_20261120.csv")
COLOUR = os.path.join(D, "colour_class_20261107.csv")
OUT = os.path.join(D, "q12_pages_manga_illust_20261122.csv")
NAME_CLS = "名前の候補(コマN・枠型)"


def main() -> int:
    o = [r for r in csv.DictReader(open(ORDER, encoding="utf-8")) if r["set"] == "main"]
    nm = {r["page_path"] for r in o if r["cand_cls"] == NAME_CLS}
    other_nm = {r["page_path"] for r in o if r["cand_cls"].startswith("名前の候補") and r["cand_cls"] != NAME_CLS}
    fin = {r["page_path"]: r for r in csv.DictReader(open(FIN, encoding="utf-8"))}
    cl = {p for p, r in fin.items() if int(r["n_清書_with_px"]) > 0}
    U = cl | nm
    print(f"(a) 清書の層(画素あり)が在る頁 {len(cl)} / (b) {NAME_CLS} の層が在る頁 {len(nm)} / 両方 {len(cl & nm)} / **和 {len(U)}**")
    print(f"  参考: cand_cls が「名前の候補」だけ(コマN・枠型でない)の頁 {len(other_nm)}、うち和の外 {len(other_nm - U)}(Q の定義どおり入れない)")
    t = {r["work_dir"]: r for r in csv.DictReader(open(TABLE, encoding="utf-8"))}
    colour = {r["page_path"]: r for r in csv.DictReader(open(COLOUR, encoding="utf-8"))}
    rows = []
    for p in sorted(U):
        f = fin[p]
        wd = f["work_dir"]
        if wd in t:
            unit, tr = "work_dir 完全一致", t[wd]
        else:
            m = [d for d in t if p.startswith(d + "\\")]
            unit, tr = ("頁の路が表のフォルダーの下", t[m[0]]) if m else ("当たらない", None)
        c = colour.get(p)
        rows.append(dict(
            page_path=p, year_label=f["year_label"], work_dir=wd,
            has_line_layer_with_px=int(p in cl), has_name_candidate=int(p in nm),
            match_unit=unit, table_work_dir=tr["work_dir"] if tr else "", table_excel_row=tr["excel_row"] if tr else "",
            table_status=tr["status"] if tr else "表に無い", answer=tr["answer"] if tr else "",
            comment=tr["comment"] if tr else "",
            colour_class_page=c["colour_class"] if c else "色の区分の表に無い",
            in_q12=int(not (tr and tr["status"] == "作者の答え" and tr["answer"] == "イラスト"))))
    print("\n当てた単位: " + " / ".join(f"{k} {v}" for k, v in collections.Counter(r["match_unit"] for r in rows).most_common()))
    print("表の値(頁):")
    for (s, a), n in sorted(collections.Counter((r["table_status"], r["answer"]) for r in rows).items()):
        print(f"  {s:6s} {a or '-':14s} {n:4d}")
    miss = [r for r in rows if r["table_status"] == "表に無い"]
    print(f"表に無い {len(miss)} 頁の色の区分: " + " / ".join(
        f"{k} {v}" for k, v in collections.Counter(r["colour_class_page"] for r in miss).items()))
    print("  フォルダー: " + " / ".join(f"{k} {v}" for k, v in sorted(collections.Counter(r["work_dir"] for r in miss).items())))
    # 表に無いフォルダーは、色の区分の表でフォルダーの全頁がモノクロか
    for w in sorted({r["work_dir"] for r in miss}):
        cc = collections.Counter(r["colour_class"] for r in colour.values() if r["work_dir"] == w)
        print(f"    {w}: フォルダーの全頁の色の区分 {dict(cc)}")
    ill = sum(1 - r["in_q12"] for r in rows)
    print(f"\n**Q12 の母集団(和 − 作者の答えがイラスト)= {len(rows)} − {ill} = {sum(r['in_q12'] for r in rows)}**")
    print("  内訳(表の値): " + " / ".join(f"{s}:{a or '-'} {n}" for (s, a), n in sorted(
        collections.Counter((r["table_status"], r["answer"]) for r in rows if r["in_q12"]).items())))
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"\n出力: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
