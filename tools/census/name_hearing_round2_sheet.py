#!/usr/bin/env python
"""**層名の聞き取り 第2回(20行)の表** —— Track E の発注(`note_census_name_hearing_round2_approved_20261112.md`、ユーザーの承認「よい」)。

**問い(第1回と違う)**: 「この名前の層は、どの頁でも『絵から除外』してよいですか」。選択肢 よい / だめ / 分からない + コメント。
**第1回(テンプレートか)の答えとは混ぜない。** 表にも答えの取り込みにも、どちらの聞き取りかを書く。

**census がすること(作る前に決めた)**:
- 行は E の下書き(`name_hearing_round2_draft_20261112.csv`)のまま。**census は行を足さない・外さない・並べ替えない**
- 下書きの数を census の側で引き直し、違えば止める
  - 名前の19行: 全層の棚卸し(`layer_name_inventory_20261110.csv`)の (層名, 種別) と、層・作品単位・作品フォルダー・頁・clip頁・lip頁の6列が全部一致すること
  - 0行目(トーン): 構造 DB の列 `tone_enabled` = 1 の層(恒久除外の頁を外す)が 91,060 であること。下書きで空欄の単位・頁の数はここで埋める
- **重なりを印字する**: 名前の19行の層のうち、`tone_enabled` = 1 でもある層の数(0行目と2回数えられる層)。答えの読み方は E が決める
- `text` 種別の行が無いことを確かめる(条例11)
- 作者に見せる列: 名前 / 種別 / 層の数 / 作品単位の数 / clip頁 / lip頁 / 答え / コメント。順位・下書きの注は見せない

    ./venv/bin/python tools/census/name_hearing_round2_sheet.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRAFT = ("inbox/name_hearing_round2_draft_20261112.csv", "b6c1f6d9598f3968")
INVENTORY = "data/census_full/layer_name_inventory_20261110.csv"
OUT_CSV = "data/census_full/name_hearing_round2_rows_20261112.csv"
OUT_XLSX = "data/census_full/hearing_layer_names_round2_20261112.xlsx"
TONE_N = 91060
TONE_LABEL = "(トーンの層 —— 名前の例: 60L 60% / 60L 10% / 30L 10% / 55L 18%)"
ANS = ["よい(どの頁でも除外してよい)", "だめ(絵のことがある)", "分からない"]
Q = "この名前の層は、どの頁でも「絵から除外」してよいですか"
COLS = ["n_layers", "n_split_units", "n_work_dirs", "n_pages", "n_clip_pages", "n_lip_pages"]


def main() -> int:
    print("[層名の聞き取り 第2回(20行)の表]")
    path, sha = DRAFT
    got = hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest()
    print(f"    下書き `{path}` sha256 {got[:16]}(便の値 {sha})")
    if not got.startswith(sha):
        return 1
    draft = list(csv.DictReader(open(os.path.join(ROOT, path), encoding="utf-8")))
    inv = {(r["layer_name"], r["v_layer_kind"]): r for r in csv.DictReader(open(os.path.join(ROOT, INVENTORY), encoding="utf-8"))}
    if any(r["v_layer_kind"] == "text" for r in draft):
        print("    **text 種別の行がある** —— 止める")
        return 1
    bad = 0
    for r in draft[1:]:
        i = inv.get((r["layer_name"], r["v_layer_kind"]))
        diff = [c for c in COLS if i is None or i[c] != r[c]]
        if diff:
            bad += 1
            print(f"    **{r['rank']} 行目が棚卸しと違う**: {diff}")
    print(f"    名前の19行: 棚卸しと6列が全部一致 {len(draft) - 1 - bad} / {len(draft) - 1}")
    if bad:
        return 1

    con = db.connect()
    ex = exclusions.load()
    un = split_units.units(con)
    names = {(r["layer_name"], r["v_layer_kind"]) for r in draft[1:]}
    rows = con.execute("""select l.page_path, coalesce(l.name, ''), k.v_layer_kind, p.format, coalesce(l.tone_enabled, false)
                            from layers l join v_layer_kind k using (page_path, layer_id) join pages p using (page_path)""").fetchall()
    tone = dict(n=0, units=set(), wds=set(), pages=set(), clip=set(), lip=set())
    kinds = collections.Counter()
    overlap = collections.Counter()
    n_ex = 0
    for p, name, kind, fmt, te in rows:
        if p in ex:
            n_ex += 1
            continue
        if not te:
            continue
        if (name, kind) in names:
            overlap[(name, kind)] += 1
        w = wd(p)
        tone["n"] += 1
        kinds[kind] += 1
        tone["units"].add(un.get(w) or "(単位なし)" + w)
        tone["wds"].add(w)
        tone["pages"].add(p)
        (tone["clip"] if fmt == "clip" else tone["lip"]).add(p)
    print(f"    0行目: tone_enabled = 1 の層 {tone['n']:,}(恒久除外の頁の層 {n_ex} を外した後)/ 便の値 {TONE_N:,} → 一致 {tone['n'] == TONE_N}")
    print(f"      種別: " + " / ".join(f"{k} {v:,}" for k, v in kinds.most_common()))
    print(f"      作品単位 {len(tone['units'])} / 作品フォルダー {len(tone['wds'])} / 頁 {len(tone['pages']):,}(clip {len(tone['clip']):,} / lip {len(tone['lip']):,})")
    if tone["n"] != TONE_N:
        return 1
    print(f"    **重なり**: 名前の19行の層のうち tone_enabled = 1 でもある層(0行目と2回数えられる)")
    for r in draft[1:]:
        k = (r["layer_name"], r["v_layer_kind"])
        if overlap[k]:
            print(f"      {r['rank']:>2} 行目({r['v_layer_kind']}): {overlap[k]:,} / {int(r['n_layers']):,}")
    print(f"      計 {sum(overlap.values()):,} 層(重なる行 {sum(1 for v in overlap.values() if v)} / 19)")

    out = []
    out.append(dict(rank=0, show_name=TONE_LABEL, layer_name="", key="tone_enabled=1", v_layer_kind="mask_fill", n_layers=tone["n"],
                    n_split_units=len(tone["units"]), n_work_dirs=len(tone["wds"]), n_pages=len(tone["pages"]),
                    n_clip_pages=len(tone["clip"]), n_lip_pages=len(tone["lip"]), n_tone_enabled=tone["n"]))
    for r in draft[1:]:
        k = (r["layer_name"], r["v_layer_kind"])
        out.append(dict(rank=int(r["rank"]), show_name=r["layer_name"], layer_name=r["layer_name"], key="name+kind", v_layer_kind=r["v_layer_kind"],
                        **{c: int(r[c]) for c in COLS}, n_tone_enabled=overlap[k]))
    with open(os.path.join(ROOT, OUT_CSV), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    write_xlsx(out)
    print(f"    → 行の表 `{OUT_CSV}` / 聞き取りの表 `{OUT_XLSX}`(非公開。層名を含む)")
    return 0


def write_xlsx(out):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    f, bo = Font(name="Arial"), Font(name="Arial", bold=True)
    fill = PatternFill("solid", fgColor="FFFF00")
    ws = wb.active
    ws.title = "説明"
    lines = [
        ("層名の聞き取り 第2回(census が作成、2026-11-12)", bo),
        ("何について答えるか: 「名前」のシートの1行が1つの層名(と種別)です。その名前の層を、どの頁でも「絵から除外」してよいかを選んでください。", f),
        ("前回(11-11)は「テンプレートか」を聞きました。今回は問いが違います。テンプレートでなくても、絵でなければ「よい」です。", bo),
        ("「よい」= どの頁のその名前の層も、絵ではないので落としてよい / 「だめ」= 頁によっては絵が描かれている / 「分からない」", f),
        ("1行目だけは名前ではなく、トーン化が効いている塗りの層をまとめた行です(名前の例: 60L 60% / 30L 10% など)。この種類の層をまとめて1つの行で聞いています。", f),
        ("種別: raster = 普通の描画の層 / mask_fill = 塗り・トーン・グラデーションの層 / folder = フォルダー", f),
        ("層の数・作品単位の数・頁の数は参考です。同じ名前が作品によって違う使われ方をしているときは、コメントに見分け方を書いてもらえると助かります。", f),
        ("黄色の列だけ記入してください。答えはプルダウン、コメントは自由記入です。", f),
        ("", f), ("記入例(このシートの中だけの例です):", bo),
        ("名前「(例)指示」/ 種別 raster / 答え → よい / コメント → 編集への指示書き", f),
    ]
    for i, (t, ft) in enumerate(lines, 1):
        ws.cell(i, 1, t).font = ft
    ws.column_dimensions["A"].width = 120
    wl = wb.create_sheet("名前")
    hdr = ["行", "層名", "種別", "層の数", "作品単位の数", ".clip の頁", ".lip の頁", Q, "コメント"]
    for j, h in enumerate(hdr, 1):
        c = wl.cell(1, j, h)
        c.font = bo
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for i, d in enumerate(out, 2):
        vals = [i - 1, d["show_name"], d["v_layer_kind"], d["n_layers"], d["n_split_units"], d["n_clip_pages"], d["n_lip_pages"], "", ""]
        for j, v in enumerate(vals, 1):
            wl.cell(i, j, v).font = f
        wl.cell(i, 8).fill = fill
        wl.cell(i, 9).fill = fill
    dv = DataValidation(type="list", formula1='"' + ",".join(ANS) + '"', allow_blank=True)
    wl.add_data_validation(dv)
    dv.add(f"H2:H{len(out) + 1}")
    for col, wdt in zip("ABCDEFGHI", (5, 40, 11, 9, 10, 9, 9, 36, 40)):
        wl.column_dimensions[col].width = wdt
    wl.row_dimensions[1].height = 45
    wl.freeze_panes = "C2"
    wb.save(os.path.join(ROOT, OUT_XLSX))


if __name__ == "__main__":
    raise SystemExit(main())
