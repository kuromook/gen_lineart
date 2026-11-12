#!/usr/bin/env python
"""**便A の聞き取りの表を、20行ずつの束に割って作る** —— Track E の返事
(`note_census_exclude_axis_and_20row_split_20261110.md` §2)と、作者が決めた聞き取りの大きさ
(2026-11-10、逐語)「excelの聞き取り表や複雑な操作が必要なUIなら２０ 画像を二択のUIなら５０ まよったら１０」。

**答えを見る前に決めたこと**:
- 頁と順番は `stage_hearing_round2_a.py` が引いたまま(14項目・再掲は8・9番目)。**並べ替えない**
- 束は E の割り方のまま: A-1 = 項目1〜5 / **A-2 = 項目6だけ(42行。1頁で20行を超える例外)** / A-3 = 7・8 / A-4 = 9・10 / A-5 = 11〜14。
  **頁の途中で束を切らない**(頁の中の層は互いに比べて答えるため)
- **最初の束 A-1 は試し**(新しい規律)。選択肢の欠けが見つかったら、直してから A-2 以降を出す
- 問いは**軸1(工程の段)だけ**。軸2(絵から除外するか)は E が後ろに回した(`note_census_template_first_hold_hearing_20261110.md`)
- 並べる層・並べ方・選択肢は 11-09 の表(`stage_hearing.py`)と同じ。コメントは自由記入
- **便A の棒は5束が全部そろってから読む**(E)。途中で読むなら「何束・何行ぶんか」を書く
- **既知セル**: 再掲の2頁は、11-09 の鍵(`hearing_drawing_stages_key_20261109.csv`)と層の並びが一致すること(一致しなければ止める)

    ./venv/bin/python tools/census/stage_hearing_round2_a_sheets.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
from stage_hearing import KINDS, OUT_KEY, STAGES, name_rule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAGES = "data/census_full/hearing_round2_a_pages_20261110.csv"
KEY_OUT = "data/census_full/hearing_round2_a_key_20261110.csv"
XLSX = "data/census_full/hearing_round2_a_bundle{b}_20261110.xlsx"
BUNDLES = [("A-1", (1, 2, 3, 4, 5)), ("A-2", (6,)), ("A-3", (7, 8)), ("A-4", (9, 10)), ("A-5", (11, 12, 13, 14))]
LIMIT = 20


def layers_top_down(rs):
    """11-09 の表と同じ並べ方: 手前から、フォルダーをたどり、raster と vector だけ"""
    by = {r[1]: r for r in rs}
    kids = collections.defaultdict(list)
    for r in rs:
        kids[r[2]].append(r)
    seq = []

    def walk(pid, path, hidden):
        for r in sorted(kids.get(pid, []), key=lambda r: -(r[3] if r[3] is not None else -1)):
            h = hidden or not (r[6] & 1)
            if r[5]:
                walk(r[1], path + ([r[4]] if r[2] is not None else []), h)
            else:
                seq.append((r, "/".join(path), h))
    for r in [r for r in rs if r[2] is None or r[2] not in by]:
        if r[5]:
            walk(r[1], [], not (r[6] & 1))
        else:
            seq.append((r, "", not (r[6] & 1)))
    out, drop, pos = [], collections.Counter(), 0
    for r, path, h in seq:
        if r[7] not in KINDS:
            drop[r[7]] += 1
            continue
        pos += 1
        out.append(dict(pos=pos, layer_id=r[1], name=r[4], folder=path,
                        shown="表示" if not h else "非表示(自身か上のフォルダー)", kind=r[7]))
    return out, drop


def main() -> int:
    print("[便A の聞き取りの表(20行ずつの束)]")
    con = db.connect()
    items = list(csv.DictReader(open(os.path.join(ROOT, PAGES), encoding="utf-8")))
    pages = [r["page_path"] for r in items]
    q = ",".join("?" * len(pages))
    ls = con.execute(f"""select l.page_path, l.layer_id, l.parent_id, l.order_in_parent, coalesce(l.name, ''),
                                l.folder, l.visibility, k.v_layer_kind
                           from layers l join v_layer_kind k using (page_path, layer_id)
                          where l.page_path in ({q})""", pages).fetchall()
    by_page = collections.defaultdict(list)
    for r in ls:
        by_page[r[0]].append(r)
    rows, drop = [], collections.Counter()
    for it in items:
        lay, d = layers_top_down(by_page[it["page_path"]])
        drop.update(d)
        if len(lay) != int(it["n_layers_raster_vector"]):
            print(f"    **項目 {it['item']} の層の数が引いた時と違う** {len(lay)} ≠ {it['n_layers_raster_vector']}")
            return 1
        for x in lay:
            rows.append(dict(item=int(it["item"]), kind_item=it["kind"], page_path=it["page_path"], wip=it["wip_status"],
                             rule=name_rule(x["name"]), **x))
    print(f"    14項目 {len(rows)}層 / 並べなかった層(フォルダーでない、種別で): "
          + " / ".join(f"{k} {v}" for k, v in drop.most_common()))
    # 既知セル: 再掲の頁は 11-09 の鍵と並びが同じ
    old = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(ROOT, OUT_KEY), encoding="utf-8")):
        old[r["page_path"]].append((int(r["pos"]), int(r["layer_id"])))
    for it in items:
        if it["kind"] != "再掲":
            continue
        new = [(x["pos"], x["layer_id"]) for x in rows if x["item"] == int(it["item"])]
        same = sorted(old[it["page_path"]]) == new
        print(f"    **既知セル** 再掲の項目 {it['item']}: 11-09 の並び {len(old[it['page_path']])}層 / 今回 {len(new)}層 / "
              + ("一致" if same else "**不一致**"))
        if not same:
            return 1
    per = collections.Counter(r["item"] for r in rows)
    with open(os.path.join(ROOT, KEY_OUT), "w", encoding="utf-8", newline="") as fh:
        bmap = {i: b for b, its in BUNDLES for i in its}
        w = csv.DictWriter(fh, fieldnames=["bundle"] + list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(dict(bundle=bmap[r["item"]], **r))
    for b, its in BUNDLES:
        sub = [r for r in rows if r["item"] in its]
        over = "  **20行を超える(1頁で切れない。E の例外)**" if len(sub) > LIMIT else ""
        print(f"    {b}: 項目 {', '.join(map(str, its))} → {len(sub)}行{over}")
        write_xlsx(b, its, items, sub, per)
    print(f"    → `{XLSX.format(b='A-1')}` ほか5本 / 鍵 `{KEY_OUT}`(非公開。名前の当て方は鍵にだけ置く)")
    return 0


def write_xlsx(b, its, items, sub, per, label="便A", n_bundles=len(BUNDLES), path=XLSX, trial=None, stages=STAGES, empty=None, machine=None, extra_lines=()):
    """`empty` を渡すと「空」の列(画素の列から機械が付けた印)を足す。便B と軸2 だけ(E 11-11。便A は変えない)。
    `machine`({(page_path, layer_id文字列): 機械の答え})を渡すと、その行の段の欄に機械の答えを書いて閉じる(灰色・プルダウンなし)。
    作者は残りの行だけに答える(ユーザー 11-12「OKだ」、E 同意)"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    f, bo = Font(name="Arial"), Font(name="Arial", bold=True)
    fill = PatternFill("solid", fgColor="FFFF00")
    grey = PatternFill("solid", fgColor="D9D9D9")
    ws = wb.active
    ws.title = "説明"
    heavy = len(sub) > LIMIT
    lines = [
        (f"工程の段の聞き取り {label} の束 {b}(Track E の依頼。census が作成、2026-11-10)", bo),
        (f"この束は {len(its)}頁・{len(sub)}行です。{label} は全部で{n_bundles}束あり、この束はそのうちの1つです。", f),
        ("何について答えるか: 「層」のシートの1行が1つの層です。その層に描かれている線が、どの段のものかを選んでください。", f),
        ("段: " + " / ".join(stages), f),
        ("黄色の列だけ記入してください(「段」はプルダウン、「コメント」は自由記入)。前回(11-09)と同じ問いです。", f),
        ("層は頁の上(手前)から順に並べています。並べたのは、フォルダーでない通常の層とベクターの層だけです。", f),
        ("選択肢に合う答えが無いときは、「段」を空けてコメントに書いてください。選択肢を直す手がかりにします。", f),
        ("迷う層は「分からない」で構いません。", f),
    ]
    if empty is not None:
        lines.append(("「空」の列は、画素を数えて機械が付けた印です(何も描かれていない層か、描いたものが自分のマスクで全部隠れている層に「空」。"
                      "11-11 に決めてもらった定義です)。空の層にも、いつもどおり段を選んでください。", f))
    for t in extra_lines:
        lines.append((t, bo))
    if machine:
        lines.append(("灰色の行は、機械が先に答えを入れた行です(何も描かれていない層 = 空 / 除外の表で確定した名前の層 = テンプレート)。"
                      "答えなくて構いません。違うと思ったら、コメントに書いてください。", bo))
    if trial if trial is not None else b == "A-1":
        lines.append(("この束は試しです。答えにくい所や選択肢の足りない所があれば、コメントに書いてください。直してから次の束を出します。", bo))
    if heavy:
        lines.append(("この束は1頁で20行を超えます(作業途中の頁です)。頁の途中で切ると層どうしを比べにくいので、例外として1頁まるごとにしました。"
                      "重すぎる・答えにくいときは、そうコメントに書いてください。", bo))
    lines += [("", f), ("記入例(このシートの中だけの例です):", bo),
              ("頁 1 / 上から 2 / 名前「レイヤー 3」/ 段 → 下絵 / コメント → 名前は付け忘れ", f)]
    for i, (t, ft) in enumerate(lines, 1):
        ws.cell(i, 1, t).font = ft
    ws.column_dimensions["A"].width = 120
    wl = wb.create_sheet("層")
    hdr = ["頁", "頁のパス", "作業途中", "上から", "層の名前", "入っているフォルダー", "表示", "種類"]
    hdr += (["空(機械の印)"] if empty is not None else []) + ["段", "コメント"]
    js = len(hdr) - 1   # 段の列の番号(1始まり)
    for j, h in enumerate(hdr, 1):
        wl.cell(1, j, h).font = bo
    wipname = {it["page_path"]: ("作業途中" if it["wip_status"] == "作業途中" else "") for it in items}
    for i, r in enumerate(sub, 2):
        vals = [r["item"], r["page_path"], wipname[r["page_path"]], r["pos"], r["name"], r["folder"], r["shown"],
                "通常" if r["kind"] == "raster" else "ベクター"]
        if empty is not None:
            vals.append("空" if empty[(r["page_path"], str(r["layer_id"]))] == "1" else "")
        mk = (machine or {}).get((r["page_path"], str(r["layer_id"])))
        vals += [("機械: " + mk) if mk else "", ""]
        for j, v in enumerate(vals, 1):
            c = wl.cell(i, j, v)
            c.font = f
            c.alignment = Alignment(vertical="top")
        if mk:
            for j in range(1, len(vals) + 1):
                wl.cell(i, j).fill = grey
            wl.cell(i, js + 1).fill = fill
        else:
            wl.cell(i, js).fill = fill
            wl.cell(i, js + 1).fill = fill
    dv = DataValidation(type="list", formula1='"' + ",".join(stages) + '"', allow_blank=True)
    wl.add_data_validation(dv)
    sc = chr(64 + js)
    for i, r in enumerate(sub, 2):
        if not (machine or {}).get((r["page_path"], str(r["layer_id"]))):
            dv.add(f"{sc}{i}")
    widths = (5, 50, 9, 7, 30, 30, 22, 9) + ((10,) if empty is not None else ()) + (34, 36)
    for col, wdt in zip("ABCDEFGHIJK", widths):
        wl.column_dimensions[col].width = wdt
    wl.freeze_panes = "E2"
    wb.save(os.path.join(ROOT, path.format(b=b)))


if __name__ == "__main__":
    raise SystemExit(main())
