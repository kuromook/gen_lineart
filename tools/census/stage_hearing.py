#!/usr/bin/env python
"""**工程の段の聞き取り表** —— Track E の発注(`note_census_request_drawing_stages_20261108.md`)への
census の提案を、ユーザーが承認した(2026-11-09「作ってもらおう」)。

**目的**: 作者に層ごとの段(あたり / 下絵 / 清書)を付けてもらい、それを照らす相手にして
**census の名前の当て方の的中率と取りこぼしを測る**(規律13)。足りなければ聞き取りの答えそのものを列にする。

**答えを見る前に決めておくこと**(結果を見て動かさない):
- **頁の引き方**: `.clip`(画素が取り出せる形式)、恒久除外を除く。名前の当て方で3つの層に分け、
  **両方(線画系と下絵系)6頁 / 下絵系だけ 3頁 / 線画系だけ 3頁**。1作品フォルダー1頁、`random.Random(0)`。
  「どちらも無い」頁は引かない(名前の当て方が何も言わない頁。取りこぼしはその外でも起こり得る、と但し書き)
- **並べる層**: フォルダーでない `raster` と `vector`(線が描かれる種別)。それ以外の種別は数えて印字する
- **名前の当て方**(作者には見せない。鍵の CSV にだけ置く): 線画系 = `線画|lineart|Lineart` →「清書」、
  下絵系 = `下絵|下書|ラフ|ﾗﾌ|rough|Rough|sketch|Sketch` →「下絵」、両方当たる名前 →「両方当たる」、どちらも無い →「なし」
- **答えから出す数**: 層ごとに 当て方 × 作者の段 の表。**清書・下絵それぞれの的中(当て方がその段と言った層のうち作者も同じ)と
  取りこぼし(作者がその段と言った層のうち当て方が言わなかった)**。頁ごとに作者が「2段以上が別の層で残る」と言った数
- **「分からない」は外して数え、外した数を印字する**
- **2026-11-09 20:23 追記(答えを見る前、ユーザーが「数が多く大変」)**: **空欄の段は「段の線ではない」と数える。**
  **集計するのは「頁」のシートの F 列(コメント)に「済」がある頁だけ**(空欄と未記入を取り違えないため)。
  頁の切れ目で止めてよい。何頁で集計したかを数の隣に印字する

    ./venv/bin/python tools/census/stage_hearing.py
"""
import collections
import csv
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_XLSX = "data/census_full/hearing_drawing_stages_20261109.xlsx"
OUT_KEY = "data/census_full/hearing_drawing_stages_key_20261109.csv"
WIP = "data/census_full/wip_pages_20261106.csv"
LINE = re.compile(r"(線画|lineart|Lineart)")
SKETCH = re.compile(r"(下絵|下書|ラフ|ﾗﾌ|rough|Rough|sketch|Sketch)")
STRATA = [("両方", 6), ("下絵系だけ", 3), ("線画系だけ", 3)]
KINDS = ("raster", "vector")
STAGES = ["あたり", "下絵", "清書(線画)", "段の線ではない(塗り・効果・資料など)", "分からない"]
PAGE_Q = "この頁に、段の違う線が別々の層として残っていますか"
PAGE_ANS = ["3段とも残っている", "2段が残っている", "1段だけ", "分からない"]


def wd(p):
    return "\\".join(p.split("\\")[:4])


def name_rule(n):
    a, b = bool(LINE.search(n)), bool(SKETCH.search(n))
    return "両方当たる" if a and b else "清書" if a else "下絵" if b else "なし"


def main() -> int:
    print("[工程の段の聞き取り表(Track E の発注、ユーザー承認 11-09)]")
    con = db.connect()
    ex = exclusions.load()
    wip = {r["page_path"]: r["wip_status"] for r in csv.DictReader(open(os.path.join(ROOT, WIP), encoding="utf-8"))}
    ls = con.execute("""select l.page_path, l.layer_id, l.parent_id, l.order_in_parent, coalesce(l.name, ''),
                               l.folder, l.visibility, k.v_layer_kind
                          from layers l join pages p using (page_path) join v_layer_kind k using (page_path, layer_id)
                         where p.format = 'clip'""").fetchall()
    by_page = collections.defaultdict(list)
    for r in ls:
        by_page[r[0]].append(r)
    n_ex = sum(p in ex for p in by_page)
    fam = {}
    for p, rs in by_page.items():
        if p in ex:
            continue
        ns = [r[4] for r in rs if not r[5] and r[7] != "text"]
        a, b = any(LINE.search(n) for n in ns), any(SKETCH.search(n) for n in ns)
        fam[p] = "両方" if a and b else "線画系だけ" if a else "下絵系だけ" if b else "どちらも無い"
    print(f"    `.clip` の頁 {len(by_page):,}(恒久除外 {n_ex} を除いて {len(fam):,})/ "
          + " / ".join(f"{k} {v:,}" for k, v in collections.Counter(fam.values()).most_common()))
    rng = random.Random(0)
    used, picks = set(), []
    for s, k in STRATA:
        cand = sorted(p for p, f in fam.items() if f == s)
        rng.shuffle(cand)
        got = 0
        for p in cand:
            if wd(p) in used:
                continue
            used.add(wd(p))
            picks.append((s, p))
            got += 1
            if got == k:
                break
        print(f"    {s}: 候補 {len(cand):,}頁(作品フォルダー {len({wd(p) for p in cand})})→ 引いた {got}")
    rows, drop = [], collections.Counter()
    for i, (s, p) in enumerate(picks, 1):
        rs = by_page[p]
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
        roots = [r for r in rs if r[2] is None or r[2] not in by]
        for r in roots:
            if r[5]:
                walk(r[1], [], not (r[6] & 1))
            else:
                seq.append((r, "", not (r[6] & 1)))
        pos = 0
        for r, path, h in seq:
            if r[7] not in KINDS:
                drop[r[7]] += 1
                continue
            pos += 1
            rows.append(dict(no=i, stratum=s, page_path=p, pos=pos, layer_id=r[1], name=r[4], folder=path,
                             shown="表示" if not h else "非表示(自身か上のフォルダー)", kind=r[7],
                             rule=name_rule(r[4]), wip=wip.get(p, "")))
    print(f"    **並べた層 {len(rows)}**(頁 {len(picks)})/ 並べなかった層(フォルダーでない、種別で): "
          + " / ".join(f"{k} {v}" for k, v in drop.most_common()))
    print("    名前の当て方(作者に見せない): " + " / ".join(f"{k} {v}" for k, v in collections.Counter(r['rule'] for r in rows).most_common()))
    per = collections.Counter(r["no"] for r in rows)
    print("    頁ごとの層: " + ", ".join(f"{i}:{per[i]}" for i in sorted(per)))
    with open(os.path.join(ROOT, OUT_KEY), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_xlsx(picks, rows)
    print(f"    → `{OUT_XLSX}` / 鍵 `{OUT_KEY}`(非公開)")
    return 0


def write_xlsx(picks, rows):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    f = Font(name="Arial")
    b = Font(name="Arial", bold=True)
    fill = PatternFill("solid", fgColor="FFFF00")
    ws = wb.active
    ws.title = "説明"
    lines = [
        ("工程の段の聞き取り(Track E の依頼。census が作成、2026-11-09)", b),
        ("何について答えるか: 「層」のシートの1行が1つの層です。その層に描かれている線が、どの段のものかを選んでください。", f),
        ("段: あたり / 下絵 / 清書(線画)/ 段の線ではない(塗り・効果・資料など)/ 分からない", f),
        ("黄色の列だけ記入してください(「段」はプルダウン、「コメント」は自由記入)。", f),
        ("層は頁の上(手前)から順に並べています。フォルダーは「入っているフォルダー」の列に書いています。", f),
        ("並べたのは、フォルダーでない通常の層とベクターの層だけです(塗りつぶし・用紙・素材・テキストなどは並べていません)。", f),
        ("迷う層は「分からない」で構いません。名前が段と合っていない層こそ、知りたいところです。", f),
        ("「頁」のシートは頁ごとに1問です(段の違う線が別々の層として残っているか)。", f),
        ("", f),
        ("記入例(このシートの中だけの例です):", b),
        ("頁 1 / 上から 12 / 名前「レイヤー 3」/ 段 → 下絵 / コメント → 名前は付け忘れ", f),
    ]
    for i, (t, ft) in enumerate(lines, 1):
        ws.cell(i, 1, t).font = ft
    ws.column_dimensions["A"].width = 120

    wp = wb.create_sheet("頁")
    hdr = ["頁", "頁のパス", "並べた層の数", "作業途中", PAGE_Q, "コメント"]
    for j, h in enumerate(hdr, 1):
        wp.cell(1, j, h).font = b
    per = collections.Counter(r["no"] for r in rows)
    for i, (s, p) in enumerate(picks, 1):
        wip = next((r["wip"] for r in rows if r["no"] == i), "")
        for j, v in enumerate([i, p, per[i], wip, "", ""], 1):
            c = wp.cell(i + 1, j, v)
            c.font = f
        wp.cell(i + 1, 5).fill = fill
        wp.cell(i + 1, 6).fill = fill
    dv = DataValidation(type="list", formula1='"' + ",".join(PAGE_ANS) + '"', allow_blank=True)
    wp.add_data_validation(dv)
    dv.add(f"E2:E{len(picks) + 1}")
    for col, wdt in zip("ABCDEF", (5, 70, 12, 10, 44, 40)):
        wp.column_dimensions[col].width = wdt

    wl = wb.create_sheet("層")
    hdr = ["頁", "頁のパス", "上から", "層の名前", "入っているフォルダー", "表示", "種類", "段", "コメント"]
    for j, h in enumerate(hdr, 1):
        wl.cell(1, j, h).font = b
    for i, r in enumerate(rows, 2):
        vals = [r["no"], r["page_path"], r["pos"], r["name"], r["folder"], r["shown"],
                "通常" if r["kind"] == "raster" else "ベクター", "", ""]
        for j, v in enumerate(vals, 1):
            c = wl.cell(i, j, v)
            c.font = f
            c.alignment = Alignment(vertical="top")
        wl.cell(i, 8).fill = fill
        wl.cell(i, 9).fill = fill
    dv2 = DataValidation(type="list", formula1='"' + ",".join(STAGES) + '"', allow_blank=True)
    wl.add_data_validation(dv2)
    dv2.add(f"H2:H{len(rows) + 1}")
    for col, wdt in zip("ABCDEFGHI", (5, 50, 7, 30, 30, 22, 9, 34, 36)):
        wl.column_dimensions[col].width = wdt
    wl.freeze_panes = "D2"
    wb.save(os.path.join(ROOT, OUT_XLSX))


if __name__ == "__main__":
    raise SystemExit(main())
