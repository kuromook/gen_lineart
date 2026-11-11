#!/usr/bin/env python
"""**第3段(テンプレートの画素の署名)の頁の一覧と、キャンバスの集計** —— Track E の発注
(`note_census_step3_corpus_order_approved_20261112.md`、ユーザーの承認「承認した」2026-11-12)と依頼
(`note_census_step3_signature_and_canvas_ask_20261112.md` §6)。

**頁の選び方(E が走る前に書いた。census は変えない)**:
- テンプレートの名前(`PICOトンボ` / `断ち落とし枠（白）` / `断ち落とし枠` / `外枠` / `ガイド`)の層を持つ `.clip` 頁のうち、
  作品単位(`split_units`)ごとに**頁のパスが辞書順で最初のもの**を1頁。恒久除外の頁は入れない
- `.lip` しか無い単位は測れないので外し、外した単位の数と名前の内訳を印字する
- 層: その頁の層を**全部**(フォルダーも。抽出側が飛ばすなら、飛ばした数を返してもらう)。census の側の列
  (`name` / `v_layer_kind` / `ancestors` / `canvas_px_w` / `canvas_px_h` / `split_unit`)を一覧に付けておく
- **先に1頁の試し**: 一覧の先頭の頁(単位の辞書順で最初)に `smoke = 1`

**キャンバスの集計**: 名前ごとに、その名前の層を持つ頁のキャンバス(`v_canvas_px_w/h`)の種類と、作品単位の数。
.clip と .lip を分ける(.lip に画素は無いが、キャンバスは構造 DB に在る)。

    ./venv/bin/python tools/census/template_pages_step3_order.py
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
NAMES = ["PICOトンボ", "断ち落とし枠（白）", "断ち落とし枠", "外枠", "ガイド"]
ORDER_OUT = "outbox/template_pages_step3_order_20261112.csv"
CANVAS_OUT = "data/census_full/template_names_canvas_20261112.csv"


def main() -> int:
    print("[第3段: テンプレートの名前の層を持つ頁の一覧と、キャンバスの集計]")
    con = db.connect()
    ex = exclusions.load()
    un = split_units.units(con)
    q = ",".join("?" * len(NAMES))
    hits = con.execute(f"""select l.page_path, l.name, k.v_layer_kind, p.format, p.v_canvas_px_w, p.v_canvas_px_h
                             from layers l join v_layer_kind k using (page_path, layer_id) join pages p using (page_path)
                            where l.name in ({q})""", NAMES).fetchall()
    n_ex = sum(h[0] in ex for h in hits)
    hits = [h for h in hits if h[0] not in ex]
    print(f"    名前の当たる層 {len(hits) + n_ex:,} → 恒久除外の頁 {n_ex} → {len(hits):,}")
    # キャンバスの集計
    cv = collections.defaultdict(lambda: collections.defaultdict(set))
    for p, n, k, f, w, h in hits:
        cv[(n, k, f)][f"{w}x{h}"].add(un[wd(p)])
    rows = []
    for (n, k, f), d in sorted(cv.items()):
        units = set().union(*d.values())
        print(f"    {n} / {k} / .{f}: 単位 {len(units)} / キャンバス {len(d)} 種 — "
              + " / ".join(f"{c} {len(u)}単位" for c, u in sorted(d.items(), key=lambda kv: -len(kv[1]))[:4])
              + (" …" if len(d) > 4 else ""))
        for c, u in d.items():
            rows.append(dict(layer_name=n, v_layer_kind=k, format=f, canvas=c, n_split_units=len(u)))
    with open(os.path.join(ROOT, CANVAS_OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # 頁の選び方
    clip = collections.defaultdict(set)
    allu = collections.defaultdict(set)
    for p, n, k, f, w, h in hits:
        allu[un[wd(p)]].add(n)
        if f == "clip":
            clip[un[wd(p)]].add(p)
    lip_only = sorted(u for u in allu if u not in clip)
    nm = collections.Counter(n for u in lip_only for n in allu[u])
    print(f"    作品単位: 名前の層を持つ {len(allu)} → .lip しか無い {len(lip_only)}(外す。名前の内訳 "
          + " / ".join(f"{a} {b}" for a, b in nm.most_common()) + f")→ 測る {len(clip)}")
    pages = sorted(min(ps) for ps in clip.values())
    qp = ",".join("?" * len(pages))
    ls = con.execute(f"""select l.page_path, l.layer_id, l.parent_id, coalesce(l.name, ''), l.folder, k.v_layer_kind,
                                p.v_canvas_px_w, p.v_canvas_px_h
                           from layers l join v_layer_kind k using (page_path, layer_id) join pages p using (page_path)
                          where l.page_path in ({qp}) order by l.page_path, l.layer_id""", pages).fetchall()
    by = {(r[0], r[1]): r for r in ls}
    out = []
    for r in ls:
        anc, pid = [], r[2]
        while pid is not None and (r[0], pid) in by:
            anc.append(by[(r[0], pid)][3])
            pid = by[(r[0], pid)][2]
        out.append([r[0], r[1], un[wd(r[0])].split("\\", 2)[-1], r[3], r[5], int(bool(r[4])), "/".join(reversed(anc)),
                    r[6], r[7], int(r[0] == pages[0])])
    with open(os.path.join(ROOT, ORDER_OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["no", "page_path", "layer_id", "split_unit", "name", "v_layer_kind", "is_folder", "ancestors",
                    "canvas_px_w", "canvas_px_h", "smoke"])
        for i, r in enumerate(out, 1):
            w.writerow([i] + r)
    nf = sum(r[5] for r in out)
    ns = [r for r in out if r[9]]
    print(f"    一覧: {len(pages)}頁 / {len(out):,}層(うちフォルダー {nf:,})/ 試しの頁: {len(ns)}層(フォルダー {sum(r[5] for r in ns)})")
    print(f"    → `{ORDER_OUT}`(非公開、抽出側への発注)/ `{CANVAS_OUT}`(非公開)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
