#!/usr/bin/env python
"""**層ごとの非表示の印** —— 作者の求め(2026-11-16「census には そのレイヤが非表示（あるいは親が非表示なので非表示もふくむ）の印のカラムはあるのか　なければ印は残してほしい」)。

**数を見る前に決めたこと**:
- 母集団: 構造 DB の全部の層(`.clip` と `.lip`、フォルダーも含む)。恒久除外の頁の層は外して数を印字する
- 列: `self_shown`(層自身の `visibility & 1`。**ビット列を等値比較しない**、T128)/ `ancestors_shown`(親を根までたどり、
  全部の祖先フォルダーの `visibility & 1` が 1 なら 1。祖先が無ければ 1)/ `shown_in_csp` = 両方 1 /
  `hidden_by`(`self` / `ancestor` / `both` / 空)
- 親の番号が同じ頁に無い層(親が見つからない)は根の直下として扱い、その数を印字する(規律9)
- **既知セル(T132: 試験対象の器具が作っていない入力)**: 抽出側の全頁の画素の表(`full_clip_pixels_layers_merged.csv`)の
  `self_shown` / `ancestors_shown` と1層ずつ突き合わせ、一致と不一致を印字する。不一致は census の値を直さずに数え、
  列 `vis_check` に `一致` / `不一致(調査中)` / `画素の表に無い` を書く(原因不明の層は『調査中』で分析対象外。作者 10-29)
- **11-16 の実測で不一致は 20層**: 抽出側が2つの層(同じファイルの写し3頁の層5・写し2頁の層167)を非表示と読み、
  構造 DB(10-26 の断面)は表示と読んでいる。20層はその2層と、その下の層。原因は census では分からない
- **言わないこと**: 「CSP の画面で見えない」の意味は表示の印だけ。不透明度 0・マスク・クリップ先の非表示・
  画面の外の配置(第2弾の12行目)は数えていない

    ./venv/bin/python tools/census/layer_visibility.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIX = "data/census_full/full_clip_pixels_layers_merged.csv"
OUT = "data/census_full/layer_visibility_20261116.csv"


def main() -> int:
    print("[層ごとの非表示の印]")
    con = db.connect()
    ex = exclusions.load()
    rows = con.execute("""select l.page_path, l.layer_id, l.parent_id, l.folder, l.visibility & 1, p.format
                            from layers l join pages p using (page_path)""").fetchall()
    n0 = len(rows)
    rows = [r for r in rows if r[0] not in ex]
    print(f"    層 {n0:,} → 恒久除外の頁の層 {n0 - len(rows)} を外して {len(rows):,}")
    by = collections.defaultdict(dict)
    for r in rows:
        by[r[0]][r[1]] = r
    orphan, loop = 0, 0
    out = []
    for p, d in by.items():
        memo = {}

        def anc(lid, seen=()):
            # 祖先(自分は含まない)が全部表示なら 1
            if lid in memo:
                return memo[lid]
            pid = d[lid][2]
            if pid is None or pid not in d or pid in seen:
                memo[lid] = 1
                return 1
            v = int(d[pid][4] == 1 and anc(pid, seen + (lid,)) == 1)
            memo[lid] = v
            return v
        for lid, r in d.items():
            if r[2] is not None and r[2] not in d:
                orphan += 1
            a, s = anc(lid), int(r[4])
            hb = {(1, 1): "", (0, 1): "self", (1, 0): "ancestor", (0, 0): "both"}[(s, a)]
            out.append(dict(page_path=p, layer_id=lid, format=r[5], is_folder=int(bool(r[3])),
                            self_shown=s, ancestors_shown=a, shown_in_csp=int(s and a), hidden_by=hb))
    print(f"    親が同じ頁に見つからない層 {orphan:,}(根の直下として扱った。親が空の層は数えない)")
    for f in ("clip", "lip"):
        for fold in (0, 1):
            s = [x for x in out if x["format"] == f and x["is_folder"] == fold]
            c = collections.Counter(x["hidden_by"] for x in s)
            print(f"    .{f} {'フォルダー' if fold else 'フォルダーでない層'} {len(s):,}: 見える {c['']:,} / 非表示 {len(s) - c['']:,}"
                  f"(層だけ {c['self']:,} / 祖先だけ {c['ancestor']:,} / 両方 {c['both']:,})")
    mine = {(x["page_path"], str(x["layer_id"])): x for x in out}
    same, diff, miss = 0, collections.Counter(), 0
    for r in csv.DictReader(open(os.path.join(ROOT, PIX), encoding="utf-8")):
        x = mine.get((r["page_path"], r["layer_id"]))
        if x is None:
            miss += 1
            continue
        k = (str(x["self_shown"]) == r["self_shown"], str(x["ancestors_shown"]) == r["ancestors_shown"])
        if all(k):
            same += 1
            x["vis_check"] = "一致"
        else:
            x["vis_check"] = "不一致(調査中)"
            diff[("層" if not k[0] else "") + ("祖先" if not k[1] else "")] += 1
    print(f"    既知セル(抽出側の全頁の画素の表): 一致 {same:,} / 不一致 {sum(diff.values()):,}"
          + ("(" + " / ".join(f"{a}が違う {b}" for a, b in diff.items()) + ")" if diff else "")
          + f" / census に無い {miss}")
    for x in out:
        x.setdefault("vis_check", "画素の表に無い")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"    → `{OUT}`(非公開、{len(out):,}行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
