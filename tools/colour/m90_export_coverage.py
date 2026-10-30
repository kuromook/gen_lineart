#!/usr/bin/env python3
"""m90: 書き出し画像が頁ごとに在るかを、名前と場所だけで数える(追補137)。

**これは棒ではない。** 署名の定義・床・帰無値・棒は「正解の範囲が決まってから登録する」と
追補130/131 で保留してある。この器具が出すのは**範囲の大きさ**だけである。

**出す列は全部「推測」である。** 抽出側の納品書が明記している:
「この画像が書き出しの画像かどうかは、判断していません。資料・背景素材・トーン素材・
下書きのスキャンなども同じ拡張子で入っています」。だから列の名前に `guess_` を付ける。
**物が何かの同定は `census` と作者の領分**(条例 項10)。

母集団は3行に分けて印字する(追補136)。合計だけの門は、両母集団がたまたま 7,663 で
等しいので 8 頁の食い違いを通してしまう。
"""
import csv, sys, json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]          # lineart-colour-manuscripts
INV_FILES = HERE / "inbox/work_exports_20261030.csv"
INV_WORKS = HERE / "inbox/work_exports_works_20261030.csv"
STRUCT    = HERE.parent / "lineart-census/inbox/request9_file_sha256_all_20261026.csv"
PIX       = sorted((HERE.parent / "lineart/inbox").glob("occ_full_p*_pages_*.csv"))
ALIAS     = HERE / "results/alias/page_alias.csv"
OUT       = HERE / "results/m90_export_coverage"

# 登録集合8頁。`022_2018_skima10_skima10_page0064` は別名表で2つに当たるので、
# **登録の「重複に乗る頁 2/8 = P0385 と P0432」で裁定した**(名前の形では決まらない)。
PILOT = ["P0151", "P0166", "P0383", "P0385", "P0432", "P0437", "P0444", "P0446"]


def norm(p):
    return p.replace("\\", "/").rstrip("/").lower()


def main():
    skipped = Counter()

    alias_of = {}
    for r in csv.DictReader(open(ALIAS, encoding="utf-8-sig")):
        alias_of[norm(r["page_path"])] = r["alias"]

    clips, lips = [], 0
    for r in csv.DictReader(open(STRUCT, encoding="utf-8-sig")):
        p = r["page_path"]
        if p.lower().endswith(".clip"):
            clips.append(p)
        elif p.lower().endswith(".lip"):
            lips += 1
            skipped[".lip(画素が無い。走行の対象外)"] += 1
        else:
            skipped["拡張子が .clip / .lip のどちらでもない"] += 1

    pix = set()
    for f in PIX:
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            pix.add(norm(r["page_path"]))

    works = {norm(r["work_dir"]): r for r in csv.DictReader(open(INV_WORKS, encoding="utf-8-sig"))}

    # 画像を「その画像が在るフォルダー」で引けるようにする
    by_dir = {}
    for r in csv.DictReader(open(INV_FILES, encoding="utf-8-sig")):
        d = norm(r["work_dir"]) + ("/" + norm(r["rel_path"]) if r["rel_path"] else "")
        by_dir.setdefault(d, []).append(r)
    n_images = sum(len(v) for v in by_dir.values())

    def owner_work(path):
        s = norm(path)
        return next((w for w in works if s.startswith(w + "/")), None)

    rows = []
    for p in clips:
        s = norm(p)
        w = owner_work(p)
        if w is None:
            skipped["作品フォルダーの外(年の直下。棚卸しの単位の外)"] += 1
            continue
        if s not in pix:
            skipped["画素の走行が触れていない(層が無い。事前登録の4頁)"] += 1
            continue
        d, _, fn = s.rpartition("/")
        stem = fn[: -len(".clip")]
        here = by_dir.get(d, [])
        same_stem = [r for r in here if r["file_name"].rsplit(".", 1)[0].lower() == stem]
        rows.append({
            "alias": alias_of.get(s, ""),
            "work_dir": works[w]["work_dir"],
            "n_clip_in_work": works[w]["n_clip"],
            "n_images_in_work": works[w]["n_images"],
            "guess_same_dir_images": len(here),
            "guess_stem_match": len(same_stem),
            "guess_stem_match_ext": "|".join(sorted({r["ext"].lower() for r in same_stem})),
            "guess_stem_match_bytes": "|".join(str(r["bytes"]) for r in same_stem),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "page_export_guess.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("== 母集団(3行に分ける。合計だけの門は 8 頁の食い違いを通す。追補136)==")
    print(f"  構造の .clip                              {len(clips):6d}")
    print(f"  画素が触れた頁                            {len(pix):6d}")
    print(f"  棚卸しの作品フォルダー / 画像ファイル     {len(works):6d} / {n_images}")
    print(f"  **採点に使える頁(交差)**                {len(rows):6d}")
    print("\n== 落とした数(理由別。規律9 / 条例 項7)==")
    for k, v in skipped.most_common():
        print(f"  {v:6d}  {k}")
    print(f"  {sum(skipped.values()):6d}  計")

    m = Counter(min(r["guess_stem_match"], 2) for r in rows)
    print("\n== 推測: 同じフォルダーに同じ名前の画像が在るか ==")
    for k in sorted(m):
        lab = {0: "0 枚", 1: "1 枚", 2: "2 枚以上"}[k]
        print(f"  {m[k]:6d} 頁 ({m[k]/len(rows):6.1%})  {lab}")
    print(f"  **在る頁 計 {len(rows)-m[0]} / {len(rows)} = {(len(rows)-m[0])/len(rows):.1%}**")

    ext = Counter()
    for r in rows:
        for e in filter(None, r["guess_stem_match_ext"].split("|")):
            ext[e] += 1
    print("\n== 推測: 同名の画像の拡張子(頁で数える。1頁に複数在りうる)==")
    for e, c in ext.most_common():
        print(f"  {c:6d} 頁  {e}")

    nodir = sum(1 for r in rows if r["guess_same_dir_images"] == 0)
    print(f"\n同じフォルダーに画像が1枚も無い頁: {nodir} ({nodir/len(rows):.1%})")
    print(f"作品フォルダーに画像が1枚も無い頁: "
          f"{sum(1 for r in rows if r['n_images_in_work']=='0')}")

    print("\n== 登録集合8頁(ここで正解が取れるかが本題)==")
    idx = {r["alias"]: r for r in rows if r["alias"]}
    for a in PILOT:
        r = idx.get(a)
        if r is None:
            print(f"  {a}  **交差に居ない**(落とした理由は上の表)")
            continue
        print(f"  {a}  同名 {r['guess_stem_match']:2d} 枚 [{r['guess_stem_match_ext'] or '—':9s}]"
              f"  同フォルダーの画像 {r['guess_same_dir_images']:4d}"
              f"  作品の画像 {r['n_images_in_work']:>5s} / 頁 {r['n_clip_in_work']:>4s}")
    have = sum(1 for a in PILOT if idx.get(a) and idx[a]["guess_stem_match"] > 0)
    print(f"  **同名の画像が在る: {have} / 8**")
    print(f"\n書いた: {OUT/'page_export_guess.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
