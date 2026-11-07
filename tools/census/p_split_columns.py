#!/usr/bin/env python
"""**Track P の依頼: 学習と評価を作品で分けるための列**(941頁)。

**依頼**: `inbox/note_p_census_columns_for_split_20261107.md`(Track P `face-merge`、2026-11-07 夜)。
頁の一覧は P の手元の `results/p2_population/candidates.csv`(941行)を読む(**P の木は書き換えない**)。

**出す列**(頁ごと、非公開。頁のパスを持つので):
- `work_dir` —— **作品フォルダー**(`F:\\manuscripts\\<年の札>\\<作品>`)。`work_id` は墨の濃さの匿名の名札(在れば)
- `arc` —— **作者の裁定の編**(2026-10-30、`arc_assignment_20261030.csv`)。**無い作品は `裁定なし`**(単独の作品という意味ではない)
- `content_links` —— **同じ中身のファイル(sha256 が同じ `.clip` / `.lip`)を共有する作品フォルダー**。
  **構造 DB の全頁で引いた**(941頁の外の頁も鎖に入る)。**作者の裁定ではなく、ファイルが同じという事実**(条例9)
- `split_unit` —— **編の裁定と、中身の共有の両方で繋がる作品フォルダーを束ねた単位**(`U<n>`)。
  **名前の似かた(`ako5` / `ako6` …)では束ねない**
- `wip_status` —— `wip_pages_20261106.csv` のまま
- `colour` —— **列が無い**(下の注)

    ./venv/bin/python tools/census/p_split_columns.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P_LIST = "../lineart-face-merge/results/p2_population/candidates.csv"
ARC = "data/census_full/arc_assignment_20261030.csv"
WORK_IDS = "data/census_full/ink_darkness_work_ids.csv"
WIP = "data/census_full/wip_pages_20261106.csv"
OUT = "data/census_full/p_split_columns_20261107.csv"
NO_COLOUR = "列なし(作者の分類を再現する列が無い。colour_ratio_vs_author.csv)"


def wd(p):
    return "\\".join(p.split("\\")[:4])


class UF:
    def __init__(self):
        self.p = {}

    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        self.p[self.f(a)] = self.f(b)


def main() -> int:
    print("[Track P の依頼: 作品で分ける列(941頁)]")
    pages = list(csv.DictReader(open(os.path.join(ROOT, P_LIST), encoding="utf-8")))
    works = sorted({wd(r["page_path"]) for r in pages})
    print(f"    頁 {len(pages)} / 作品フォルダー {len(works)}")
    arc = {(r["year_label"], r["work"]): r["arc"]
           for r in csv.DictReader(open(os.path.join(ROOT, ARC), encoding="utf-8"))}
    wid = {r["work_dir"]: r["work_id"] for r in csv.DictReader(open(os.path.join(ROOT, WORK_IDS), encoding="utf-8"))}
    wip = {r["page_path"]: r["wip_status"] for r in csv.DictReader(open(os.path.join(ROOT, WIP), encoding="utf-8"))}
    con = db.connect()
    by_hash, by_path_hash = collections.defaultdict(set), collections.defaultdict(set)
    for p, s in con.execute("select page_path, f_sha256 from file_hashes where f_sha256 is not null").fetchall():
        by_hash[s].add(wd(p))
        by_path_hash[s].add(p)
    arc_of = {}
    for (y, n), a in arc.items():
        arc_of[f"F:\\manuscripts\\{y}\\{n}"] = a
    # **編の境を跨ぐ中身の共有は束ねない** —— 作者が別の編に置いたフォルダーを、ファイルが同じだからと
    # 繋ぐと作者の分類を上書きする(条例9)。**跨いだ組は数えて印字する**
    content, both = UF(), UF()
    crossed = collections.Counter()
    crossed_in_p = 0
    pset = {r["page_path"] for r in pages}
    for h, ws in by_hash.items():
        ws = sorted(ws)
        for w in ws[1:]:
            content.u(ws[0], w)
            a, b = arc_of.get(ws[0]), arc_of.get(w)
            if a and b and a != b:
                crossed[(ws[0].split("\\", 2)[2], a, w.split("\\", 2)[2], b)] += 1
                continue
            both.u(ws[0], w)
    for h, ps in by_path_hash.items():
        if {wd(p) for p in ps} and len({arc_of.get(wd(p)) for p in ps} - {None}) > 1 and ps & pset:
            crossed_in_p += 1
    for (wa, a, wb, b), n in sorted(crossed.items()):
        print(f"    **編の境を跨ぐ中身の共有(束ねなかった)**: {wa}({a})— {wb}({b}) ファイル {n}")
    print(f"    うち Track P の 941頁が含まれるファイル {crossed_in_p}")
    first = {}
    for w, a in sorted(arc_of.items()):
        if a in first:
            both.u(first[a], w)
        else:
            first[a] = w
    groups_c, groups_b = collections.defaultdict(set), collections.defaultdict(set)
    for w in set(content.p) | set(works) | set(arc_of):
        groups_c[content.f(w)].add(w)
        groups_b[both.f(w)].add(w)
    unit_name, out, missing_wip = {}, [], 0
    for w in works:
        r = both.f(w)
        if r not in unit_name:
            unit_name[r] = f"U{len(unit_name) + 1}"
    for r in pages:
        w = wd(r["page_path"])
        links = sorted(groups_c[content.f(w)] - {w})
        s = wip.get(r["page_path"])
        missing_wip += s is None
        out.append([r["page_path"], w, wid.get(w, ""), arc_of.get(w, "裁定なし"),
                    ";".join(x.split("\\", 2)[2] for x in links), unit_name[both.f(w)], s or "", NO_COLOUR])
    n_unit = collections.Counter(x[5] for x in out)
    print(f"    **編の裁定がある頁 {sum(x[3] != '裁定なし' for x in out)}**"
          f"(作品フォルダー {len({x[1] for x in out if x[3] != '裁定なし'})})")
    print(f"    **中身を共有する別の作品フォルダーがある頁 {sum(bool(x[4]) for x in out)}**"
          f"(作品フォルダー {len({x[1] for x in out if x[4]})})")
    print(f"    **分ける単位 {len(n_unit)}**(作品フォルダー {len(works)} から)/ 頁の多い順 "
          + ", ".join(f"{u} {n}" for u, n in n_unit.most_common(5)))
    multi = [u for u in n_unit if len({x[1] for x in out if x[5] == u}) > 1]
    print(f"    **2つ以上の作品フォルダーを束ねた単位 {len(multi)}**")
    for u in sorted(multi, key=lambda u: -n_unit[u]):
        fs = sorted({x[1].split("\\", 2)[2] for x in out if x[5] == u})
        print(f"      {u}: 頁 {n_unit[u]} / {fs}")
    print(f"    作業途中 {sum(x[6] == '作業途中' for x in out)} / 裁定なし {sum(x[6] == '裁定なし' for x in out)}"
          f" / wip の表に無い頁 {missing_wip}")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["page_path", "work_dir", "work_id", "arc", "content_links", "split_unit", "wip_status", "colour"])
        w.writerows(out)
    print(f"    → `{OUT}`(非公開)")
    return 1 if missing_wip else 0


if __name__ == "__main__":
    raise SystemExit(main())
