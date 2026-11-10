#!/usr/bin/env python
"""**制作の方式**(アナログ / デジタル)を、作者の裁定のまま台帳に載せる。

**出所は作者の記入式の表**(2026-10-31、Track N 経由)。**track が推測した列ではない。**
Track N の依頼(逐語): **「上の表をそちらの台帳に載せてほしい。(…)
他の track が同じ推測を別々に作るのを止めたい」**。
**それがこの木の役割である**(ユーザー、2026-10-28: 原稿の情報の正本は census)。

**この表の形について、作者の答えが1つ決めたことが在る**:

> **下絵は11作すべてアナログ。分散がゼロ。**

**だから「アナログ/デジタル」を1列で持ってはいけない** —— **2段が同じ札に潰れる。**
列は **`stage_rough`(下絵)** と **`stage_lineart`(線画)** に分ける。
**画素に効くのは線画の段階である**(線画の層からタイルを切るため)。

**この器具が届く範囲**(名前より狭く書く。T132):

- **届く**: 書かれた (年の札, 作品) が**母集団に実在するか** /
  **便が添えた編が、編の表(作者の裁定 2026-10-30)と一致するか** /
  その作品フォルダーの頁数と層数
- **届かない**: **方式そのもの。** **作者の言明は測定ではない**(条例6)——
  画素と一致して見えても、一致であって確認ではない。
  そして**「時期の効果」としては読めない** —— **アナログ線画の3作はすべて 2024 系の札に
  固まっているので、工程と時期が一緒に動く**(Track N も同じ壁に当たっている)

    ./venv/bin/python tools/census/author_process.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402
import db as census_db  # noqa: E402
import keys as census_keys  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT = "data/census_full/author_process_20261031.csv"
# **2026-11-10 追加**: 試しの20作の答え(作者が直接記入。表 `process_hearing_pilot20.py`)。**値は書き換えない** ——
# 線画の「分からない」は、コメント「線画はない」と並べて載せる(選択肢に「線画が無い」が無かった = 表の不具合)
PILOT = "data/census_full/hearing_process_pilot20_answered_20261110.xlsx"
ARC_READBACK = "inbox/work_arcs_20261030.csv"


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def work_dir_of(page_path: str, year: str, work: str):
    """**頁の道から作品フォルダーの道を切り出す。** 組み立てない。

    組み立てた道は、置き場が変わった日に**黙って別の鍵を配る** ——
    鍵が動く表は、引用されたときに別の物を指す(`keys.py`)。
    """
    needle = f"\\{year}\\{work}\\"
    i = page_path.find(needle)
    if i < 0:
        return None
    return page_path[:i + len(needle) - 1]


NOTE = [
    "**2026-11-10 に20行を足した**(試しの20作。作者が census の表に直接記入、Track N の依頼)。"
    "**下の 10-31 の注は、10-31 の11作についての言明のまま残す**(書き換えない)",
    "**試しの20作の線画「分からない」12行のうち11行は、作者のコメントが「線画はない」**"
    "(作者、2026-11-10:「今回は線画のない鉛筆漫画原稿がおおい」)。**表の選択肢に「線画が無い」が無かった** —— "
    "表の不具合で、作者の迷いではない。**値は書き換えず、コメントと並べて載せる。「分からない」を「無い」と読み替えるのは読む側の仕事**",
    "**試しの20作は 19作である** —— 2行(札が違う同じ名前の作品)は作者が「同じ内容」と答えた(2026-11-10)。"
    "census の分ける単位でも、ファイルの中身の共有で同じ単位に束ねてあった",
    "**出所は作者の記入式の表**(2026-10-31、Track N 経由)。"
    "**census は作者に直接聞いていない。** **track が推測した列ではない** —— "
    "**他の track が同じ推測を別々に作るのを止めるために、ここが正本を持つ**"
    "(ユーザー、2026-10-28)",
    "**下絵と線画は別の列である。** **下絵は11作すべてアナログで、分散がゼロ** —— "
    "**1列で持つと2段が同じ札に潰れる**(作者はいつも紙で下描きしている)。"
    "**画素に効くのは線画の段階**である(線画の層から画素を取るため)",
    "**「時期の効果」として読めない。** **アナログ線画の3作はすべて 2024 系の札に"
    "固まっている**ので、**工程と時期が一緒に動く。** "
    "**単調な「デジタル化」でもない** —— 2017〜2023b は線画デジタルで、"
    "**2024 でアナログ線画に戻っている**",
    "**`device_note` の「未記載」を「iPad ではない」と読まない。** "
    "作者は**4作にだけ `ipad で線画` と書いた** —— 残りは**書かれていない**のであって、"
    "**否定されたのではない**(条例6 の家族)。**機材は工程と別の軸である**",
    "**仕上げの列は無い。** 作者は「アナログの仕上げはまずない」と言い、"
    "**在れば備考に書く約束だったが、どの行にも書かれていない** —— "
    "**「0件だった」ではなく「列を立てていない」**",
    "**10-31 の11作は母集団 320 組のうちの 11 組である。** "
    "**残りに方式の裁定は無い** —— **空欄は「デジタル」ではない**(条例4 の家族)",
]


def pilot_rows(arcs):
    """試しの20作の答え。作者が census の表に直接記入した(Track N の依頼)"""
    from openpyxl import load_workbook
    ws = load_workbook(os.path.join(ROOT, PILOT), read_only=True)["作品"]
    it = ws.iter_rows(values_only=True)
    h = list(next(it))
    out = []
    for v in it:
        r = dict(zip(h, v))
        if r["番号"] is None:
            continue
        k = (str(r["札"]), r["作品"])
        out.append(dict(year_label=k[0], work=k[1], arc_from_n=arcs.get(k, "未割り当て"),
                        stage_rough=r["下絵"] or "未記入", stage_lineart=r["線画"] or "未記入",
                        device_note=r["機材の備考"] or "未記載", comment=r["コメント"] or "",
                        decided_by="作者の記入式の表(census が作成、直接)", as_of="2026-11-10",
                        source_notice="hearing_process_pilot20_answered_20261110.xlsx"))
    return out


def main() -> int:
    print("  [制作の方式(作者の裁定)]")
    arcs = {(r["year"], r["work"]): r["arc"] for r in load(ARC_READBACK)}
    rows_in = [dict(r, comment="") for r in load(INPUT)] + pilot_rows(arcs)
    con = census_db.connect()
    try:
        pages = collections.Counter()
        layers = collections.Counter()
        for y, w, n in con.execute(
                "select year, work, count(*) from pages group by 1,2").fetchall():
            pages[(y, w)] = n
        for y, w, n in con.execute(
                """select p.year, p.work, count(*) from layers l
                   join pages p using(page_path) group by 1,2""").fetchall():
            layers[(y, w)] = n
        n_works_pop = con.execute(
            "select count(*) from (select distinct year, work from pages)").fetchone()[0]
        # **作品フォルダーの道は組み立てずに材料から取る** ——
        # 組み立てた道は、置き場が変わった日に黙って別の鍵を配る
        sample = {(y, w): p for y, w, p in con.execute(
            "select year, work, min(page_path) from pages group by 1,2").fetchall()}
    finally:
        con.close()

    # **落ちたものを理由別に数えて印字する**(規律9)
    drops = collections.Counter()
    kept = []
    for r in rows_in:
        k = (r["year_label"], r["work"])
        if k not in pages:
            drops["母集団の (年の札, 作品) に無い"] += 1
            continue
        kept.append(r)
    print(f"    裁定 {len(rows_in)} 組 / 母集団に在る {len(kept)}"
          f" —— **母集団の組は {n_works_pop}**(**残りに方式の裁定は無い**)")
    for why, n in sorted(drops.items()):
        print(f"      落とした {n} 組 —— {why}")
    if drops:
        print("    **実在しない組に裁定が付いている。表を書かずに止める。**")
        return 1

    # **便が添えた編と、編の表(作者の裁定 2026-10-30)を突き合わせる。**
    # **一致しても「正しい」ではなく「転記が合っている」**(T132)
    differ = unassigned = 0
    for r in kept:
        k = (r["year_label"], r["work"])
        a = arcs.get(k)
        if a is None:
            unassigned += 1
            if r["arc_from_n"] != "未割り当て":
                differ += 1
        elif a != r["arc_from_n"]:
            differ += 1
    print(f"    編の表(42組)との突き合わせ: 編が付いている {len(kept) - unassigned}"
          f" / 編の表に無い {unassigned} / **食い違い {differ}**")
    if differ:
        print("    **便の編と編の表が食い違っている。表を書かずに止める。**")
        return 1

    reg = census_keys.work_registry()
    rows = []
    for r in sorted(kept, key=lambda r: (r["stage_lineart"], r["year_label"],
                                         r["work"])):
        k = (r["year_label"], r["work"])
        work_dir = work_dir_of(sample[k], r["year_label"], r["work"])
        if work_dir is None:
            print(f"    **作品フォルダーの道が頁の道から取れない組が在る。止める。**")
            return 1
        rows.append([reg.key_for(work_dir, where=os.path.basename(INPUT)),
                     r["year_label"],
                     arcs.get(k, "**未裁定**"),
                     r["stage_rough"], r["stage_lineart"], r["device_note"],
                     pages[k], layers[k],
                     r["decided_by"], r["as_of"], r["source_notice"], r["comment"], ""])
    reg.report("作品の鍵")
    reg.close()

    by = collections.Counter((r[3], r[4]) for r in rows)
    print("    方式の内訳(**下絵 / 線画**):")
    for (rough, line), n in sorted(by.items()):
        print(f"      下絵 {rough} / 線画 {line}: {n} 組")
    dev = collections.Counter(r[5] for r in rows)
    print("    機材: " + " / ".join(f"{k} {v}" for k, v in sorted(dev.items())))

    rows += [["", "", "", "", "", "", "", "", "", "", "", "", n] for n in NOTE]
    header = ["work_id", "year_label", "arc", "stage_rough", "stage_lineart",
              "device_note", "n_pages", "n_layers", "by", "as_of", "source", "author_comment", "note"]
    n = bc.write_csv(os.path.join(ROOT, "census", "author_process.csv"), header, rows)
    print(f"    author_process.csv {n}行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
