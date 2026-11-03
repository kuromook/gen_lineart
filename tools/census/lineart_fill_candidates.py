#!/usr/bin/env python
"""**名前が線画なのに塗りつぶしの層の候補**を選んで、作者に聞く表を作る(2026-11-03、ユーザーの依頼)。

**きっかけ**: sketch set の聞き取り(11-03 22:37)で、**名前が線画の層のうち2層が頁全体の塗りつぶし**だった
(墨の濃さの主結論の母集団の墨画素の 2.4%)。**同じものが sketch set の外にも在るかもしれない**(ユーザー)。

**この器具は同定しない**(規律13・条例9)。**篩(ふるい)で候補を選び、何であるかは作者に聞く。**

- **篩**: 描かれた画素(`alpha > 0`)÷ キャンバスの画素(`n_px_total`。抽出側: キャンバスの画素数)。
  **境は作者が仕上げの線と答えた層の最大値より上**(既知の陰性の最大。器具が印字する)。
  **境はデータから引いた** —— **陽性は2層しか無く、篩の取りこぼしは測れない**(取りこぼしは書けない、と書く)
- **母集団**: 墨の濃さ 1段目・2段目の層別 L1 形式2(主結論の母集団 40,323層 + 1段目)
- **出すもの**: `data/census_full/hearing_lineart_fill_candidates_20261103.xlsx`(**原稿の名前を含むので非公開**)。
  **公開の表は答えが来てから**

    ./venv/bin/python tools/census/lineart_fill_candidates.py
"""
import collections
import csv
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ink_darkness as ik  # noqa: E402
import worksheet_folder_reach as wf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_XLSX = "data/census_full/hearing_lineart_fill_candidates_20261103.xlsx"
ANS = ["線画(線だけ)", "線画とベタ(線の層にベタも塗った)", "塗りつぶし", "その他", "分からない"]
Q = "この層は何ですか"


def load_rows():
    rows = [dict(r, ink_stage="1") for r in csv.DictReader(open(os.path.join(ROOT, ik.STAGE1), encoding="utf-8"))]
    for p in sorted(glob.glob(os.path.join(ROOT, ik.STAGE2_GLOB))):
        rep = ik.STAGE2_REPLACED.get(os.path.basename(p))
        if rep:
            p = os.path.join(os.path.dirname(p), rep)
        rows += [dict(r, ink_stage="2") for r in csv.DictReader(open(p, encoding="utf-8"))]
    return rows


def cover(r):
    t = int(r["n_px_total"] or 0)
    return int(r["n_px_alpha_gt0"] or 0) / t if t else 0.0


def main() -> int:
    print("  [名前が線画なのに塗りつぶしの層の候補]")
    import duckdb
    con = duckdb.connect(wf.DB, read_only=True)
    keys = {(p, int(i)) for p, i, _ in wf.inside_keys(con)}
    hearing = wf.load_hearing()
    ids = {r["work_dir"]: r["work_id"] for r in
           csv.DictReader(open(os.path.join(ROOT, wf.WORK_IDS), encoding="utf-8"))}
    drops = collections.Counter()
    pop = []
    for r in load_rows():
        if r["stratum"] != "L1":
            continue
        if r["pixel_formats"] != "2":
            drops["形式2 でない(主結論の母集団の外)"] += 1
            continue
        if not int(r["n_px_total"] or 0):
            drops["キャンバスの画素数が 0"] += 1
            continue
        pop.append(r)
    for why, n in sorted(drops.items()):
        print(f"    数えなかった層 {n:,} —— {why}")

    def ins(r):
        return (r["page_path"], int(r["layer_id"])) in keys
    pos = [r for r in pop if ins(r) and hearing.get(r["page_path"]) in wf.HEARING_EXCLUDE]
    neg = [r for r in pop if ins(r) and hearing.get(r["page_path"]) == "仕上げの線"]
    edge = max(cover(r) for r in neg)
    print(f"    **既知**: 作者が塗りつぶしと答えた頁の層 {len(pos)}(被覆 "
          + ", ".join(f"{cover(r):.3f}" for r in pos) + f")/ 仕上げの線と答えた頁の層 {len(neg):,}(被覆の最大 {edge:.4f})")
    cand = sorted((r for r in pop if cover(r) > edge and not (r in pos)), key=lambda r: -cover(r))
    total_ink = sum(int(r["n_px_ink"] or 0) for r in pop)
    print(f"    **候補(被覆 > {edge:.4f}、既知の2層を除く)**: {len(cand)}層 / "
          f"墨画素の {sum(int(r['n_px_ink'] or 0) for r in cand) / total_ink:.1%}(母集団 {len(pop):,}層)")
    print(f"    うち sketch set の外 {sum(not ins(r) for r in cand)} / 中 {sum(ins(r) for r in cand)}")
    print("    段ごと: " + " / ".join(f"{k}段目 {v}" for k, v in sorted(collections.Counter(r["ink_stage"] for r in cand).items())))
    print(f"    **被覆 0.99 以上**: {sum(cover(r) >= 0.99 for r in cand)}層")
    write_xlsx(cand, ids, ins, total_ink, edge, len(neg))
    print(f"    聞き取りの表 `{OUT_XLSX}`(非公開)")
    return 0


def write_xlsx(cand, ids, ins, total_ink, edge, n_neg):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation
    F = "Arial"
    base, bold, big = Font(name=F, size=10), Font(name=F, size=10, bold=True), Font(name=F, size=12, bold=True)
    head_fill, fill_in = PatternFill("solid", fgColor="D9D9D9"), PatternFill("solid", fgColor="FFFF00")
    thin = Side(style="thin", color="BFBFBF")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap = Alignment(wrap_text=True, vertical="top")
    wb = Workbook()
    s0 = wb.active
    s0.title = "説明"
    lines = [
        ("名前が「線画」なのに塗りつぶしかもしれない層の聞き取り", big),
        (f"作成: {time.strftime('%Y-%m-%d %H:%M')}(census track)。非公開(data/ 配下。commit しない)", base),
        ("", base),
        ("聞きたいこと", bold),
        ("sketch set の聞き取りで、名前が「線画」の層のうち2層が頁全体の塗りつぶしでした(W10 の page0026 の双子)。", base),
        ("同じような層が sketch set の外にもあるかを確かめたいので、キャンバスに対して描かれた面積が大きい「線画」の層を選びました。", base),
        (f"選び方: 描かれた画素 ÷ キャンバスの画素 が {edge:.3f} より大きい層。"
         f"{edge:.3f} は、前回「仕上げの線」と答えていただいた頁の、sketch set の中の線画の層({n_neg:,}層)の最大値です。", base),
        ("面積が大きいだけで塗りつぶしとは限りません(ベタを線の層に塗った頁など)。何の層かを教えてください。", base),
        ("", base),
        ("埋め方", bold),
        (f"・シート「層」の黄色のセルだけ埋めてください。問い:「{Q}」。選択肢はプルダウンです。合わなければ自由に書いて構いません。", base),
        ("・「頁(ファイルの場所)」で原稿を開き、「層の名前」と「層の番号(layer_id)」で層を探してください。", base),
        ("・同じ頁の層が並んでいるときは、層ごとに答えてください。", base),
        ("", base),
        ("答えの使い方", bold),
        ("・「塗りつぶし」と答えた層は、墨の濃さで「作者の答えで外した値」から外します(主値 0.9177 は動かしません)。", base),
        ("・「線画とベタ」「線画」は数に入れたままにします。", base),
        ("", base),
        ("記入例(1行。値は架空)", bold),
    ]
    for i, (t, f) in enumerate(lines, 1):
        s0.cell(row=i, column=1, value=t).font = f
    r = len(lines) + 1
    for j, (h, v) in enumerate(zip(["頁(ファイルの場所)", "層の名前", Q, "コメント"],
                                   ["F:\\manuscripts\\20xx\\(作品名)\\page0001.clip", "線画", "線画とベタ(線の層にベタも塗った)",
                                    "髪のベタを線画の層に直接塗っている"]), 1):
        a = s0.cell(row=r, column=j, value=h)
        a.font, a.fill, a.border, a.alignment = bold, head_fill, box, wrap
        b = s0.cell(row=r + 1, column=j, value=v)
        b.font, b.border, b.alignment = base, box, wrap
    r += 3
    s0.cell(row=r, column=1, value="進み具合(自動)").font = bold
    s0.cell(row=r + 1, column=1, value="答えを埋めた層の数").font = base
    s0.cell(row=r + 1, column=2, value=f"=COUNTA(層!F2:F{len(cand) + 1})").font = base
    s0.cell(row=r + 2, column=1, value="層の数").font = base
    s0.cell(row=r + 2, column=2, value=f"=COUNTA(層!B2:B{len(cand) + 1})").font = base
    for col, w in (("A", 70), ("B", 30), ("C", 30), ("D", 40)):
        s0.column_dimensions[col].width = w

    ws = wb.create_sheet("層")
    head = ["作品フォルダー", "頁(ファイルの場所)", "層の名前", "層の番号(layer_id)", "sketch set の中か",
            Q, "コメント", "描かれた画素 ÷ キャンバス", "描かれた墨のうち純黒", "不透明度(256 が 100%)",
            "母集団の墨に占める割合", "札", "census の名札(匿名・照合用)"]
    for j, h in enumerate(head, 1):
        c = ws.cell(row=1, column=j, value=h)
        c.font, c.fill, c.border, c.alignment = bold, head_fill, box, wrap
    for i, r in enumerate(cand, 2):
        ink = int(r["n_px_ink"] or 0)
        vals = [r["work_dir"], r["page_path"], r["layer_name"], int(r["layer_id"]), "中" if ins(r) else "外",
                None, None, cover(r), (int(r["n_px_pure_black"] or 0) / ink if ink else None),
                int(r["layer_opacity"] or 0), ink / total_ink, r["year_label"], ids.get(r["work_dir"], "")]
        for j, v in enumerate(vals, 1):
            c = ws.cell(row=i, column=j, value=v)
            c.font, c.border, c.alignment = base, box, wrap
            if j in (6, 7):
                c.fill = fill_in
        ws.cell(row=i, column=8).number_format = "0.0%"
        ws.cell(row=i, column=9).number_format = "0.0%"
        ws.cell(row=i, column=11).number_format = "0.00%"
    for j, w in enumerate([26, 52, 18, 10, 8, 26, 40, 12, 12, 10, 10, 7, 10], 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.row_dimensions[1].height = 48
    ws.freeze_panes = "C2"
    dv = DataValidation(type="list", formula1='"' + ",".join(ANS) + '"', allow_blank=True, showErrorMessage=False)
    ws.add_data_validation(dv)
    dv.add(f"F2:F{len(cand) + 1}")
    wb.save(os.path.join(ROOT, OUT_XLSX))


if __name__ == "__main__":
    raise SystemExit(main())
