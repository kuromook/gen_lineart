#!/usr/bin/env python
"""**二重に入っている頁の聞き取りの表(フォルダーの組の単位、15行)** —— 基盤の回送(ユーザーの決定「二重に入ってるページのどれを使うかはcensus」)。

**なぜフォルダーの組で聞くか**: `duplicate_pages_v5.py` の組(同じ 年\\作品名 で同じファイル名の頁が別のフォルダーにある)は 114組・257頁あるが、
フォルダーの組に畳むと 15、組どうしのつながりで畳むと 8 になる。しかも**組の中には、版違いではなく別の作品・別の巻が同じファイル名を持つだけのもの**
(例: 一つのフォルダーに複数の作品が入った束)が混ざって見える。**どちらかは名前や統計量で決めず、作者に聞く**(2026-10-26)。
1行 = つながったフォルダーの組(組どうしが同じフォルダーを共有するときは1行に畳む)。作者の聞き取りの大きさ「Excel は 20」に収まる。**最初の表は試し**。

**答えの選択肢(作る前に決めた)**: 版違い(採るフォルダーを書く)/ 別の作品・別の巻(二重ではない)/ 分からない。
採るフォルダーは自由記入(フォルダーの最後の部分の名前)。

    ./venv/bin/python tools/census/hearing_duplicate_folders_v5.py
"""
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = "data/census_full/duplicate_pages_v5_20261112.csv"
KEY_OUT = "data/census_full/hearing_duplicate_folders_v5_key_20261112.csv"
XLSX = "data/census_full/hearing_duplicate_folders_v5_20261112.xlsx"
ANS = ["版違い(同じ頁。採るフォルダーを右に書く)", "別の作品・別の巻(二重ではない。全部使う)", "分からない"]
Q = "これらのフォルダーの同じ名前のファイルは、同じ頁の版違いですか"


def main() -> int:
    print("[二重の頁の聞き取りの表(フォルダーの組)]")
    rows = list(csv.DictReader(open(os.path.join(ROOT, SRC), encoding="utf-8")))
    g = collections.defaultdict(list)
    for r in rows:
        g[r["group"]].append(r)
    # フォルダーの組が重なる(同じフォルダーが別の組にも出る)ので、つながったフォルダーを1行に畳む(union-find)
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for rs in g.values():
        fs = sorted({r["folder"] for r in rs})
        for x in fs[1:]:
            par[find(x)] = find(fs[0])
    comp = collections.defaultdict(list)
    for rs in g.values():
        comp[find(rs[0]["folder"])].append(rs)
    items = []
    for _, groups in sorted(comp.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        fs = tuple(sorted({r["folder"] for rs in groups for r in rs}))
        same = sum(1 for rs in groups if len({r["content_fingerprints"] for r in rs}) == 1)
        files = sorted({rs[0]["file"] for rs in groups})
        items.append(dict(folders=fs, n_groups=len(groups), n_pages=sum(len(rs) for rs in groups), n_same_content=same,
                          examples=", ".join(files[:3]) + (" …" if len(files) > 3 else "")))
    print(f"    フォルダーの組(重なりあり){len({tuple(sorted({r['folder'] for r in rs})) for rs in g.values()})} → つながりで畳んで {len(items)}")
    print(f"    組 {len(g)} → フォルダーの組 {len(items)}")
    with open(os.path.join(ROOT, KEY_OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "folders", "n_groups", "n_pages", "n_same_content"])
        for i, it in enumerate(items, 1):
            w.writerow([i, " | ".join(it["folders"]), it["n_groups"], it["n_pages"], it["n_same_content"]])
    write_xlsx(items)
    print(f"    → 鍵 `{KEY_OUT}` / 表 `{XLSX}`(非公開)")
    return 0


def write_xlsx(items):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    f, bo = Font(name="Arial"), Font(name="Arial", bold=True)
    fill = PatternFill("solid", fgColor="FFFF00")
    ws = wb.active
    ws.title = "説明"
    lines = [
        ("二重に入っている頁の聞き取り(census が作成、2026-11-12)", bo),
        ("下絵と線画の対(v5)の中に、同じ作品の同じ名前のファイルが、別のフォルダーから2つ以上入っています(114組・257頁)。", f),
        ("何について答えるか: 「組」のシートの1行が、同じ名前のファイルを持つフォルダーの組み合わせです。"
         "そのフォルダーどうしの同じ名前のファイルが、同じ頁の版違い(作業途中のコピーなど)か、別の作品・別の巻がたまたま同じ名前なのかを選んでください。", f),
        ("版違いのときは、使うフォルダーの名前(パスの最後の部分)を「採るフォルダー」に書いてください。頁ごとに違うとき、フォルダーが3つ以上で一部だけ版違いのときは、コメントに書いてください。", f),
        ("「中身が同じ組」は、対の画像の指紋が全く同じ組の数です。参考で、答えを決めるものではありません。", f),
        ("黄色の列だけ記入してください。答えはプルダウン、採るフォルダーとコメントは自由記入です。", f),
        ("この表は試しです。答えにくい所や選択肢の足りない所があれば、コメントに書いてください。", bo),
        ("", f), ("記入例(このシートの中だけの例です):", bo),
        ("フォルダー「(例)作品\\done3」「(例)作品\\done4」/ 答え → 版違い / 採るフォルダー → done4 / コメント → done4 が完成版", f),
    ]
    for i, (t, ft) in enumerate(lines, 1):
        ws.cell(i, 1, t).font = ft
    ws.column_dimensions["A"].width = 120
    wl = wb.create_sheet("組")
    hdr = ["行", "フォルダー1", "フォルダー2", "フォルダー3", "フォルダー4", "同じ名前のファイルの組", "うち中身が同じ組", "ファイルの例", Q, "採るフォルダー", "コメント"]
    for j, h in enumerate(hdr, 1):
        c = wl.cell(1, j, h)
        c.font = bo
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for i, it in enumerate(items, 2):
        fs = [x.replace("F:\\manuscripts\\", "") for x in it["folders"]] + ["", "", ""]
        if len(it["folders"]) > 4:
            raise SystemExit("フォルダーが5つ以上の行がある(列を足す)")
        vals = [i - 1, fs[0], fs[1], fs[2], fs[3], it["n_groups"], it["n_same_content"], it["examples"], "", "", ""]
        for j, v in enumerate(vals, 1):
            wl.cell(i, j, v).font = f
        for j in (9, 10, 11):
            wl.cell(i, j).fill = fill
    dv = DataValidation(type="list", formula1='"' + ",".join(ANS) + '"', allow_blank=True)
    wl.add_data_validation(dv)
    dv.add(f"I2:I{len(items) + 1}")
    for col, wdt in zip("ABCDEFGHIJK", (5, 28, 28, 28, 28, 10, 10, 30, 40, 16, 36)):
        wl.column_dimensions[col].width = wdt
    wl.row_dimensions[1].height = 45
    wl.freeze_panes = "B2"
    wb.save(os.path.join(ROOT, XLSX))


if __name__ == "__main__":
    raise SystemExit(main())
