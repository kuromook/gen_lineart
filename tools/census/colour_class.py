#!/usr/bin/env python
"""**カラーかモノクロか**の列 —— 作者の決め方(2026-11-07)の第1段と、第2段の聞き取りの表。

**作者の決め方(逐語、11-07)**:「カラーレイヤが全くないものはモノクロで確定である　カラーレイヤがあるもののうち
聞き取りできめるしかないのかもしれない　…　まず 漫画作品とイラスト用のフォルダで分け、漫画作品ならどのファイルが
カラーかを聞き取る　イラストフォルダはほとんどがカラーだが、ラフで終わってるもの、モノクロイラストというケースが
わずかにあり、それらが聞き取り対象」。**数えない層の裁定(11-07)**:「用紙、トーンの層、下絵の層は、表現色がカラーでも
カラーの層に数えない　あってる / 素材は数えない、素材は大抵カラーだから / まずこの戦略でやってみてほしい」。

**カラーの層** = フォルダーでない層のうち、**表現色がカラー(`colour_type_index = 0`)** で、次のどれでもないもの:
用紙(`v_layer_kind = paper`)/ トーン(`tone_enabled`)/ 素材(`v_layer_kind = image_material`)/
**下絵(名前に `下絵` を含む)** —— **下絵の層を名前で引くのは census の当て方**(作者は「下絵の層」と言った)。
**表示・非表示を問わず数える**(作者「全くない」。非表示の層だけを持つ頁の数も印字する)。

**第1段の値**: `モノクロ(カラーの層なし)` / `聞き取り待ち(カラーの層あり)`。**後者をカラーと読まない。**

**既知セル**: 作者がカラー/モノクロを裁定した頁(`colour_pilot.AUTHOR_CLASS`)。**ただし数えない層の規則はこの頁を
見てから立てた** —— **同じ頁で確かめても裏付けにならない**(印字はするが、独立の既知セルではない)。

    ./venv/bin/python tools/census/colour_class.py
"""
import collections
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402
import colour_pilot as cp  # noqa: E402
import db  # noqa: E402
import exclusions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_FULL = "data/census_full/colour_class_20261107.csv"
OUT_PUB = "census/colour_class.csv"
OUT_XLSX = "data/census_full/hearing_work_folder_manga_or_illust_20261107.xlsx"
P_LIST = "../lineart-face-merge/results/p2_population/candidates.csv"
MONO = "モノクロ(カラーの層なし)"
ASK = "聞き取り待ち(カラーの層あり)"
ANS = ["漫画", "イラスト", "漫画とイラストが混ざる", "その他", "分からない"]
Q = "この作品フォルダーは漫画ですか、イラストですか"


def wd(p):
    return "\\".join(p.split("\\")[:4])


def main() -> int:
    print("[カラーかモノクロか(作者の決め方の第1段)]")
    con = db.connect()
    excl = exclusions.load()
    rows = con.execute("""
        select p.page_path, p.format, p.year,
          count(l.layer_id) filter (where c) as n_colour,
          count(l.layer_id) filter (where c and (l.visibility & 1) = 1) as n_colour_vis,
          count(l.layer_id) filter (where l.colour_type_index = 0 and not l.folder) as n_ct0,
          count(l.layer_id) filter (where l.colour_type_index = 0 and not l.folder and k.v_layer_kind = 'paper') as d_paper,
          count(l.layer_id) filter (where l.colour_type_index = 0 and not l.folder and k.v_layer_kind <> 'paper'
                                    and coalesce(l.tone_enabled, false)) as d_tone,
          count(l.layer_id) filter (where l.colour_type_index = 0 and not l.folder and k.v_layer_kind = 'image_material'
                                    and not coalesce(l.tone_enabled, false)) as d_material,
          count(l.layer_id) filter (where l.colour_type_index = 0 and not l.folder
                                    and k.v_layer_kind not in ('paper', 'image_material')
                                    and not coalesce(l.tone_enabled, false)
                                    and coalesce(l.name, '') like '%下絵%') as d_sketch
        from pages p
        left join (select l.*, k.v_layer_kind,
                          (l.colour_type_index = 0 and not l.folder
                           and k.v_layer_kind not in ('paper', 'image_material')
                           and not coalesce(l.tone_enabled, false)
                           and coalesce(l.name, '') not like '%下絵%') as c
                     from layers l join v_layer_kind k using (page_path, layer_id)) l using (page_path)
        left join v_layer_kind k using (page_path, layer_id)
        group by all""").fetchall()
    n_ex = sum(r[0] in excl for r in rows)
    rows = [r for r in rows if r[0] not in excl]
    print(f"    頁 {len(rows):,}(恒久除外の頁 {n_ex} を除いた)")
    d = collections.Counter()
    for r in rows:
        d["表現色カラーの層(フォルダーでない)"] += r[5]
        d["数えない: 用紙"] += r[6]
        d["数えない: トーン"] += r[7]
        d["数えない: 素材"] += r[8]
        d["数えない: 名前に下絵"] += r[9]
        d["カラーの層"] += r[3]
    for k, v in d.items():
        print(f"      {k} {v:,}")
    if d["表現色カラーの層(フォルダーでない)"] != d["カラーの層"] + sum(v for k, v in d.items() if k.startswith("数えない")):
        print("    **数えない層の内訳が合わない**")
        return 1
    cls = {r[0]: (MONO if r[3] == 0 else ASK) for r in rows}
    for fmt in ("clip", "lip"):
        s = [r for r in rows if r[1] == fmt]
        print(f"    {fmt}: 頁 {len(s):,} / **{MONO} {sum(cls[r[0]] == MONO for r in s):,}** / "
              f"{ASK} {sum(cls[r[0]] == ASK for r in s):,}"
              f"(うちカラーの層が全部非表示 {sum(r[3] > 0 and r[4] == 0 for r in s)})")
    # 既知セル(独立ではない)
    hit = collections.Counter()
    for p, c in cp.AUTHOR_CLASS.items():
        if p in cls:
            hit[(c, cls[p])] += 1
    print("    **作者の裁定の頁**(数えない層の規則はこの頁を見てから立てた。独立の既知セルではない):")
    for (c, v), n in sorted(hit.items()):
        print(f"      {c} → {v}: {n}")
    bad = sum(n for (c, v), n in hit.items() if c == "カラー" and v == MONO)
    if bad:
        print(f"    **作者がカラーとした頁をモノクロにした {bad}**")

    # 作品フォルダー
    pset = set()
    if os.path.exists(os.path.join(ROOT, P_LIST)):
        pset = {r["page_path"] for r in csv.DictReader(open(os.path.join(ROOT, P_LIST), encoding="utf-8"))}
    w = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        a = w[wd(r[0])]
        a["n"] += 1
        a["mono"] += cls[r[0]] == MONO
        a["ask"] += cls[r[0]] == ASK
        a["p"] += r[0] in pset
        a["p_ask"] += r[0] in pset and cls[r[0]] == ASK
        a[r[1]] += 1
    need = {k: v for k, v in w.items() if v["ask"]}
    print(f"    作品フォルダー {len(w)} / **全頁がモノクロに決まった {len(w) - len(need)}** / "
          f"**聞き取りが要る(カラーの層のある頁を持つ){len(need)}**(うち Track P の頁を持つ "
          f"{sum(1 for v in need.values() if v['p_ask'])})")
    print(f"    Track P の 941頁: {MONO} {sum(cls.get(p) == MONO for p in pset)} / {ASK} {sum(cls.get(p) == ASK for p in pset)}")

    with open(os.path.join(ROOT, OUT_FULL), "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["page_path", "format", "work_dir", "colour_class", "n_colour_layers", "n_colour_layers_visible",
                     "n_ct0_layers", "n_skip_paper", "n_skip_tone", "n_skip_material", "n_skip_sketch_name"])
        for r in rows:
            wr.writerow([r[0], r[1], wd(r[0]), cls[r[0]], r[3], r[4], r[5], r[6], r[7], r[8], r[9]])
    by_year = collections.defaultdict(collections.Counter)
    for r in rows:
        y = r[0].split("\\")[2]
        by_year[(y, r[1])]["n"] += 1
        by_year[(y, r[1])][cls[r[0]]] += 1
    pub = [[y, f, v["n"], v[MONO], v[ASK], ""] for (y, f), v in sorted(by_year.items())]
    pub.append(["ALL", "", len(rows), sum(c == MONO for c in cls.values()), sum(c == ASK for c in cls.values()), ""])
    pub += [[""] * 5 + [n] for n in NOTE]
    n_pub = bc.write_csv(os.path.join(ROOT, OUT_PUB),
                         ["year_label", "format", "n_pages", "n_pages_mono_certain", "n_pages_colour_layer", "note"], pub)
    write_xlsx(need)
    print(f"    {OUT_PUB} {n_pub}行 / 頁ごと `{OUT_FULL}` / 聞き取りの表 `{OUT_XLSX}`(非公開)")
    return 1 if bad else 0


def write_xlsx(need):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
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
        ("作品フォルダーは漫画か、イラストか", big),
        (f"作成: {time.strftime('%Y-%m-%d %H:%M')}(census track)。非公開(data/ 配下。commit しない)", base),
        ("", base),
        ("聞きたいこと", bold),
        ("カラーの層(表現色カラー。用紙・トーン・下絵・素材は数えない)が1枚も無い頁はモノクロに決めました。", base),
        ("残りの頁を決めるために、まず作品フォルダーが漫画かイラストかを教えてください(あなたの決め方の第2段)。", base),
        ("ここに並べたのは、カラーの層がある頁を1頁以上持つ作品フォルダーだけです。全頁がモノクロに決まったフォルダーは聞きません。", base),
        ("", base),
        ("埋め方", bold),
        (f"・シート「作品フォルダー」の黄色のセルだけ埋めてください。問い:「{Q}」。選択肢はプルダウンです。", base),
        ("・「Track P」の列に数があるフォルダーは Track P が使う頁を持っています。先に埋めてもらえると P が先に進めます。", base),
        ("", base),
        ("答えの使い方", bold),
        ("・「漫画」: 次に、そのフォルダーのどのファイルがカラーかを聞きます(表紙など)。", base),
        ("・「イラスト」: カラーとして扱い、ラフで止めたもの・モノクロのイラストだけを次に聞きます。", base),
        ("・「漫画とイラストが混ざる」「その他」「分からない」: コメントを読んで決めます。", base),
        ("", base),
        ("記入例(1行。値は架空)", bold),
        ("F:\\manuscripts\\20xx\\(作品名) / 漫画 / 表紙だけ別ファイル", base),
        ("", base),
        ("進み具合(自動)", bold),
    ]
    for i, (t, f) in enumerate(lines, 1):
        s0.cell(row=i, column=1, value=t).font = f
    r = len(lines) + 1
    s0.cell(row=r, column=1, value="答えを埋めたフォルダーの数").font = base
    s0.cell(row=r, column=2, value=f"=COUNTA(作品フォルダー!G2:G{len(need) + 1})").font = base
    s0.cell(row=r + 1, column=1, value="フォルダーの数").font = base
    s0.cell(row=r + 1, column=2, value=f"=COUNTA(作品フォルダー!A2:A{len(need) + 1})").font = base
    s0.column_dimensions["A"].width = 110
    ws = wb.create_sheet("作品フォルダー")
    head = ["作品フォルダー", "頁", "うち .lip", "モノクロに決まった頁", "カラーの層がある頁", "Track P の頁(カラーの層がある)",
            Q, "コメント"]
    for j, h in enumerate(head, 1):
        c = ws.cell(row=1, column=j, value=h)
        c.font, c.fill, c.border, c.alignment = bold, head_fill, box, wrap
    order = sorted(need.items(), key=lambda kv: (-(kv[1]["p_ask"] > 0), kv[0]))
    for i, (k, v) in enumerate(order, 2):
        vals = [k, v["n"], v["lip"], v["mono"], v["ask"], v["p_ask"] or None, None, None]
        for j, x in enumerate(vals, 1):
            c = ws.cell(row=i, column=j, value=x)
            c.font, c.border, c.alignment = base, box, wrap
            if j in (7, 8):
                c.fill = fill_in
    for j, wdt in enumerate([40, 7, 7, 10, 10, 12, 22, 40], 1):
        ws.column_dimensions[chr(64 + j)].width = wdt
    ws.row_dimensions[1].height = 48
    ws.freeze_panes = "B2"
    dv = DataValidation(type="list", formula1='"' + ",".join(ANS) + '"', allow_blank=True, showErrorMessage=False)
    ws.add_data_validation(dv)
    dv.add(f"G2:G{len(need) + 1}")
    wb.save(os.path.join(ROOT, OUT_XLSX))


NOTE = [
    "**作者の決め方(2026-11-07)の第1段**: **カラーの層が1枚も無い頁はモノクロに確定**(作者)。"
    "**カラーの層 = フォルダーでない層で、表現色がカラー(`colour_type_index = 0`)。ただし用紙・トーン・下絵・素材は数えない**(作者の裁定 11-07)",
    "**`n_pages_colour_layer` はカラーの頁の数ではない** —— **カラーの層を持つ頁で、カラーかどうかは聞き取りで決める**"
    "(作者: まず作品フォルダーを漫画とイラストに分け、漫画ならどのファイルがカラーか、イラストならラフ止まり・モノクロのものを聞く)",
    "**下絵の層は名前に `下絵` を含む層として引いた**(census の当て方)。**表示・非表示を問わず数えた**(作者「全くない」)",
    "**作者がカラー/モノクロを裁定した頁で、カラーをモノクロにした頁は 0。ただし数えない層の規則はその頁を見てから立てたので、独立の確かめではない**",
    "**頁・作品の名前は書かない**(条例10)。**頁ごとの全量は `data/census_full/` 側**",
]

if __name__ == "__main__":
    raise SystemExit(main())
