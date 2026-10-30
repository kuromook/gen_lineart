#!/usr/bin/env python3
"""m91: 届いた指紋を棚卸しに継ぎ、**寸法だけで候補を絞れるか**を測る(追補139)。

**これは棒ではない。** 署名の定義・床・帰無値・棒は保留のまま(追補130/131)。
ここで出すのは「寸法という鍵がどこまで効くか」の見積もりである。

**恒等の門は頁ごとに当てる**(条例 項3 / T158): 合計が 11,549 で合っても、鍵が1対1で
当たっているとは限らない。だから**鍵の重なりを数えて印字する。**

**`status = ok` でも使えない指紋が在る**(納品書 §3): 合成が全面白の PSD 32 枚、
`convert("RGBA")` で頭打ちになる 16bit PNG 4 枚。**理由別に数えて落とす**(規律9)。
"""
import csv, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
INV  = HERE / "inbox/work_exports_20261030.csv"
SIG  = HERE / "inbox/work_export_signatures_20261030.csv"
COV  = HERE / "results/m10_base_set/set_coverage.csv"
ALIAS= HERE / "results/alias/page_alias.csv"
GUESS= HERE / "results/m90_export_coverage/page_export_guess.csv"
OUT  = HERE / "results/m91_sig_dims"

WHITE = "ff" * 1024
PILOT = ["P0151", "P0166", "P0383", "P0385", "P0432", "P0437", "P0444", "P0446"]


def key(r):
    return (r["work_dir"].lower(), r["rel_path"].lower(), r["file_name"].lower())


def main():
    inv = {key(r): r for r in csv.DictReader(open(INV, encoding="utf-8-sig"))}
    sig = {}
    dup = 0
    for r in csv.DictReader(open(SIG, encoding="utf-8-sig")):
        k = key(r)
        if k in sig:
            dup += 1
        sig[k] = r

    print("== 鍵の恒等の門(合計ではなく重なりを数える。条例 項3 / T158)==")
    print(f"  棚卸しの行 {len(inv)} / 指紋の行 {len(sig)} / 指紋側の鍵の重複 {dup}")
    print(f"  両方に在る鍵        {len(inv.keys() & sig.keys())}")
    print(f"  棚卸しにだけ在る鍵  {len(inv.keys() - sig.keys())}")
    print(f"  指紋にだけ在る鍵    {len(sig.keys() - inv.keys())}")

    st = Counter(r["status"] for r in sig.values())
    print("\n== status(納品書と一致するか)==")
    for k, v in st.most_common():
        print(f"  {v:6d}  {k}")

    unusable = Counter()
    usable = {}
    for k, r in sig.items():
        if r["status"] != "ok":
            unusable[f"status = {r['status']}"] += 1
        elif r["sig32"] == WHITE:
            # **理由で割る**(規律9)。混ぜると「器具の都合」と「中身が白」が1つの数になる。
            # 納品書が挙げたのは前の2つで、3つ目は**本当に白い画像**(3D のテクスチャの
            # AO 図など)。**合計 71 を1行で書くと、納品書の 32 + 4 と合わないように見える。**
            if r["file_name"].lower().endswith(".psd"):
                unusable["指紋が全面白 / PSD の合成が白(器具の都合。納品書 §3)"] += 1
            elif r["mode"] == "I;16":
                unusable["指紋が全面白 / 16bit の頭打ち(器具の都合。納品書 §4)"] += 1
            else:
                unusable["指紋が全面白 / 中身が白い画像(器具の都合ではない)"] += 1
        elif not r["sig32"]:
            unusable["sig32 が空なのに status = ok"] += 1
        else:
            usable[k] = r
    print("\n== 使えない指紋(理由別。規律9)==")
    for k, v in unusable.most_common():
        print(f"  {v:6d}  {k}")
    print(f"  {sum(unusable.values()):6d}  計   → **使える指紋 {len(usable)} / {len(sig)}**")

    # 登録集合8頁の画布
    alias_of, canvas_of = {}, {}
    rows = list(csv.DictReader(open(COV, encoding="utf-8-sig")))
    for r in csv.DictReader(open(ALIAS, encoding="utf-8-sig")):
        alias_of[r["alias"]] = r["page_path"]
    name_canvas = {r["page"]: r["canvas"] for r in rows}
    # results の頁名 → 別名 は m90 の出力が持っている
    g = {r["alias"]: r for r in csv.DictReader(open(GUESS, encoding="utf-8-sig")) if r["alias"]}

    # 頁名と別名の対応は「年 + 作品の語 + 幹」で引く(m90 と同じ裁定)
    stem_of = {
        "P0151": ("2017", "page0002"), "P0166": ("2017", "page0019"),
        "P0383": ("2018", "page0003"), "P0385": ("2018", "page0005"),
        "P0432": ("2018", "page0064"), "P0437": ("2018", "page0073"),
        "P0444": ("2018", "skima12h"), "P0446": ("2018", "skima12chara"),
    }
    canvas_for = {}
    for a, (yr, stem) in stem_of.items():
        hit = [n for n, c in name_canvas.items() if n.split("_")[1] == yr and n.lower().endswith("_" + stem)]
        canvas_for[a] = name_canvas[hit[0]] if len(hit) == 1 else None

    OUT.mkdir(parents=True, exist_ok=True)
    out_rows = []
    print("\n== 登録集合8頁: 寸法だけでどこまで絞れるか ==")
    print("   頁    画布        候補  寸法一致  比一致(±1%)  使える指紋")
    for a in PILOT:
        gi = g.get(a)
        if gi is None:
            print(f"  {a}  **交差に居ない**")
            continue
        w = gi["work_dir"].lower()
        cand = [r for k, r in sig.items() if k[0] == w]
        cv = canvas_for[a]
        if cv is None:
            print(f"  {a}  画布が引けない(頁名の対応が一意でない)")
            continue
        W, H = (int(x) for x in cv.split("x"))
        exact, aspect, ok = [], [], []
        for r in cand:
            if r["status"] != "ok" or not r["width"]:
                continue
            ok.append(r)
            iw, ih = int(r["width"]), int(r["height"])
            if (iw, ih) == (W, H):
                exact.append(r)
            if ih and abs(iw / ih - W / H) / (W / H) <= 0.01:
                aspect.append(r)
        print(f"  {a}  {cv:11s} {len(cand):5d} {len(exact):8d} {len(aspect):12d} "
              f"{sum(1 for r in ok if r['sig32'] and r['sig32']!=WHITE):10d}")
        for r in exact[:4]:
            out_rows.append({"alias": a, "canvas": cv, "match": "exact",
                             "rel_path": r["rel_path"], "file_name": r["file_name"],
                             "width": r["width"], "height": r["height"], "mode": r["mode"]})
    if out_rows:
        with open(OUT / "pilot_dim_candidates.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            wr.writeheader(); wr.writerows(out_rows)
        print(f"\n書いた: {OUT/'pilot_dim_candidates.csv'}({len(out_rows)} 行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
