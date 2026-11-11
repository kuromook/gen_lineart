#!/usr/bin/env python
"""**a(決定論の除外)の第1歩: テンプレートの層名の候補を出し、作者に固定してもらう表を作る**
—— ユーザーの決定(2026-11-11、a/b/c の分け方。a = census)。

**ユーザーの言葉**: 「PICOトンボや枠などのテンプレートは全く同じレイヤでしかない　レイヤ名もぶれていない
これらは決定論的に除外できるはずだ」(11-10、E の便から)/「a.トンボや空のレイヤは決定論的に判定出来る」(11-11)

**候補の選び方(結果を見る前に決めた)** —— 画素は使わない(構造 DB だけ。画素の署名は承認の後の発注):
- 構造の署名 = (層名, 種別 `v_layer_kind`, `layer_type`, `colour_type_index`, `blend_mode`, `opacity`, `offset_x`, `offset_y`,
  キャンバスの px 寸法, 親の名前)。**テンプレートは型のファイルから写すので、作品をまたいで署名が完全に一致するはず**
- 名前ごとに、**最も多くの作品単位にまたがる署名**を1つ取り、その単位の数で並べる(多い順)
- **一致の割合** = その署名の層 ÷ その名前の層。テンプレートなら 1 に近いはず(「ぶれていない」)。並べ替えには使わず、表に並べる
- **上位20の名前を聞く**(作者の聞き取りの大きさ「Excel は 20」)。恒久除外の頁は外す。`text` 種別は外す(名前が本文)
- 名前はぶれを畳まない(完全一致)

**この表は作者に聞くための候補で、除外の表ではない。** 作者が「テンプレート」と答えた名前だけを、のちに除外の名前の表に入れる
(許可表と同じ考え方。census は自分で足さない)。

    ./venv/bin/python tools/census/template_candidates.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_ASK = 20
OUT_CSV = "data/census_full/template_candidates_20261111.csv"
OUT_XLSX = "data/census_full/hearing_template_names_20261111.xlsx"
ANS = ["テンプレート(絵ではない。どの頁でも除外してよい)", "テンプレートではない", "一部だけテンプレート", "分からない"]
Q = "この名前の層は、型から写したテンプレートですか"


def main() -> int:
    print("[テンプレートの層名の候補(a: 決定論の除外)]")
    con = db.connect()
    ex = exclusions.load()
    un = split_units.units(con)
    rows = con.execute("""
        select l.page_path, coalesce(l.name, ''), k.v_layer_kind, l.layer_type, l.colour_type_index, l.blend_mode,
               l.opacity, l.offset_x, l.offset_y, p.v_canvas_px_w, p.v_canvas_px_h, coalesce(par.name, ''), p.format
          from layers l join v_layer_kind k using (page_path, layer_id) join pages p using (page_path)
          left join layers par on par.page_path = l.page_path and par.layer_id = l.parent_id""").fetchall()
    n_ex = n_text = 0
    by_name = collections.defaultdict(lambda: collections.defaultdict(lambda: dict(n=0, units=set(), pages=set(), lip=0)))
    tot = collections.Counter()
    for r in rows:
        if r[0] in ex:
            n_ex += 1
            continue
        if r[2] == "text":
            n_text += 1
            continue
        sig = r[2:12]
        a = by_name[r[1]][sig]
        a["n"] += 1
        a["units"].add(un[wd(r[0])])
        a["pages"].add(r[0])
        a["lip"] += r[12] == "lip"
        tot[r[1]] += 1
    print(f"    層 {len(rows):,} → 外した: 恒久除外の頁 {n_ex} / テキスト {n_text:,} → 名前 {len(by_name):,}")
    best = []
    for name, sigs in by_name.items():
        sig, a = max(sigs.items(), key=lambda kv: (len(kv[1]["units"]), kv[1]["n"]))
        best.append(dict(name=name, kind=sig[0], n_units_top_sig=len(a["units"]), n_layers_top_sig=a["n"],
                         n_pages_top_sig=len(a["pages"]), n_lip_top_sig=a["lip"], n_layers_name=tot[name],
                         share_top_sig=round(a["n"] / tot[name], 3), n_signatures=len(sigs), parent=sig[9],
                         opacity=sig[4], offset=f"{sig[5]},{sig[6]}", canvas=f"{sig[7]}x{sig[8]}"))
    best.sort(key=lambda d: (-d["n_units_top_sig"], -d["n_layers_top_sig"], d["name"]))
    for i, d in enumerate(best, 1):
        d["rank"] = i
    with open(os.path.join(ROOT, OUT_CSV), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank"] + [k for k in best[0] if k != "rank"])
        w.writeheader()
        w.writerows(best)
    ask = best[:N_ASK]
    print(f"    上位 {N_ASK} を聞く(作品単位の数の多い順。並び替えに一致の割合は使わない):")
    for d in ask:
        print(f"      {d['rank']:>2} 単位 {d['n_units_top_sig']:>3} / 層 {d['n_layers_top_sig']:>5} / 一致の割合 {d['share_top_sig']:.3f} "
              f"/ 署名 {d['n_signatures']} / {d['kind']}")
    print(f"      21位の単位の数 {best[N_ASK]['n_units_top_sig']}(聞かない側の先頭)")
    write_xlsx(ask)
    print(f"    → 全候補 `{OUT_CSV}`({len(best):,}行)/ 聞き取りの表 `{OUT_XLSX}`(非公開。層名を含む)")
    return 0


def write_xlsx(ask):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    f, bo = Font(name="Arial"), Font(name="Arial", bold=True)
    fill = PatternFill("solid", fgColor="FFFF00")
    ws = wb.active
    ws.title = "説明"
    lines = [
        ("テンプレートの層名の聞き取り(a: 決定論の除外。census が作成、2026-11-11)", bo),
        ("何について答えるか: 「名前」のシートの1行が1つの層名です。その名前の層が、型のファイルから写したテンプレート"
         "(トンボ・断ち落とし枠など。絵ではない)かを選んでください。", f),
        ("「テンプレート」と答えた名前だけを、除外の名前の表に入れます。census は自分では足しません。", f),
        ("並べたのは、同じ名前・同じ設定(種別・不透明度・位置・キャンバス寸法・親の名前)の層が、最も多くの作品にまたがっている名前の上位20です。", f),
        ("「一致の割合」は、その名前の層のうち、いちばん多い設定と全く同じ層の割合です(1 に近いほどぶれていない)。", f),
        ("黄色の列だけ記入してください。答えはプルダウン、コメントは自由記入です。", f),
        ("一部だけテンプレートのとき(同じ名前で絵の層もある等)は、コメントに見分け方を書いてもらえると助かります。", f),
        ("同じ設定かどうかは画素を見ずに比べたので、トンボのような型の層でも一致の割合は 0.4 くらいでした(キャンバスの寸法や親が作品で違うため)。"
         "そのため、作業の手順で使う層の名前も上位に入っています。違うものは「テンプレートではない」で構いません。", f),
        ("この表は試しです。答えにくい所や選択肢の足りない所があれば、コメントに書いてください。", bo),
        ("", f), ("記入例(このシートの中だけの例です):", bo),
        ("名前「(例)枠線」/ 答え → テンプレート / コメント → 印刷所の型", f),
    ]
    for i, (t, ft) in enumerate(lines, 1):
        ws.cell(i, 1, t).font = ft
    ws.column_dimensions["A"].width = 120
    wl = wb.create_sheet("名前")
    hdr = ["順", "層名", "種類", "親の名前", "作品の数(同じ設定)", "層の数(同じ設定)", "うち .lip", "その名前の層の数", "一致の割合", Q, "コメント"]
    for j, h in enumerate(hdr, 1):
        c = wl.cell(1, j, h)
        c.font = bo
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for i, d in enumerate(ask, 2):
        vals = [d["rank"], d["name"] or "(名前なし)", d["kind"], d["parent"] or "(最上位)", d["n_units_top_sig"], d["n_layers_top_sig"], d["n_lip_top_sig"],
                d["n_layers_name"], d["share_top_sig"], "", ""]
        for j, v in enumerate(vals, 1):
            wl.cell(i, j, v).font = f
        wl.cell(i, 10).fill = fill
        wl.cell(i, 11).fill = fill
    dv = DataValidation(type="list", formula1='"' + ",".join(ANS) + '"', allow_blank=True)
    wl.add_data_validation(dv)
    dv.add(f"J2:J{len(ask) + 1}")
    for col, wdt in zip("ABCDEFGHIJK", (5, 26, 12, 18, 10, 10, 8, 10, 8, 36, 36)):
        wl.column_dimensions[col].width = wdt
    wl.row_dimensions[1].height = 45
    wl.freeze_panes = "C2"
    wb.save(os.path.join(ROOT, OUT_XLSX))


if __name__ == "__main__":
    raise SystemExit(main())
