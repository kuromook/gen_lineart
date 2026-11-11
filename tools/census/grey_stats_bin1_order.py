#!/usr/bin/env python
"""**階調の統計の便1 の層の一覧**(Track Q の登録 `q1`、E の起案 §1。ユーザーの承認「便１の発注を承認」2026-11-11)。

**引く前に決めたこと**(Q `note_q_census_order_grey_stats_bin1_approved_20261111.md` / Q の案
`note_q_e_census_hello_b_and_kuro_keshi_registered_20261111.md` §3。点数は無い。census は一覧を引くだけ):
- 全系統で `.clip` だけ(`.lip` は画素を読む道具が無い)。恒久除外の頁は入れない(`exclusions`)
- **黒消しの材料A**: 名前 `成人修正` かつ種別 raster の全層
- **黒消しの材料B**: 名前 `モザイク` の frame_folder の下の葉の層(フォルダーでない層)から 50。作品単位ごとに均等
- **対照 用紙**: 種別 paper から 50。作品単位ごとに均等
- **対照 コマ枠**: frame_folder(名前 `モザイク` を含むものを除く)そのものから 50。作品単位ごとに均等。
  **抽出側が枠そのものを層として描けるかは分からない** —— 描けなければ「試していない」(Q が印字)
- **対照 髪や服のベタ**: 名前がちょうど `ベタ` の raster の層(`線画+ベタ` などの混ざった名前は入れない) + 作者のコメントに `ベタ` が在る層(層の役割の表 v2)。全部。
  50 に足りなくても**名前で埋めない**(Q「引けない対照は無理に埋めず」)
- 均等の引き方: 作品単位を `random.Random(20261111)` で並べ、各単位の候補も同じ乱数で並べ、単位を順に1層ずつ取る(round-robin)
- 系統が重なる層は先の系統に置き、後の系統では飛ばす(数を印字)

    ./venv/bin/python tools/census/grey_stats_bin1_order.py
"""
import collections
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 20261111
N_PER = 50
TABLE = "data/census_full/layer_role_table_20261110_v2.csv"
OUT = "outbox/grey_stats_bin1_layers_20261111.csv"
KURO, MOZA, BETA = "成人修正", "モザイク", "ベタ"


def even(cands, un, rng, n):
    by = collections.defaultdict(list)
    for k in sorted(cands):
        by[un[wd(k[0])]].append(k)
    us = sorted(by)
    rng.shuffle(us)
    for u in us:
        rng.shuffle(by[u])
    out = []
    while len(out) < n and any(by[u] for u in us):
        for u in us:
            if by[u] and len(out) < n:
                out.append(by[u].pop())
    return out


def main() -> int:
    print(f"[階調の統計 便1 の層の一覧] seed {SEED}")
    con = db.connect()
    ex = exclusions.load()
    un = split_units.units(con)
    rows = con.execute("""select l.page_path, l.layer_id, l.parent_id, coalesce(l.name, ''), l.folder, k.v_layer_kind, p.format
                            from layers l join v_layer_kind k using (page_path, layer_id) join pages p using (page_path)""").fetchall()
    by = {(r[0], r[1]): r for r in rows}
    drop = collections.Counter()

    def ok(r):
        if r[6] != "clip":
            drop[".lip"] += 1
            return False
        if r[0] in ex:
            drop["恒久除外の頁"] += 1
            return False
        return True

    def under_moza(r):
        pid = r[2]
        while pid is not None and (r[0], pid) in by:
            a = by[(r[0], pid)]
            if a[5] == "frame_folder" and a[3] == MOZA:
                return True
            pid = a[2]
        return False

    rng = random.Random(SEED)
    sel = {}
    kuro = [(r[0], r[1]) for r in rows if r[3] == KURO and r[5] == "raster" and ok(r)]
    sel["黒消しの材料A(名前)"] = sorted(kuro)
    moza = [(r[0], r[1]) for r in rows if not r[4] and under_moza(r) and ok(r)]
    sel["黒消しの材料B(モザイクの中の葉)"] = even(moza, un, rng, N_PER)
    paper = [(r[0], r[1]) for r in rows if r[5] == "paper" and ok(r)]
    sel["対照 用紙"] = even(paper, un, rng, N_PER)
    koma = [(r[0], r[1]) for r in rows if r[5] == "frame_folder" and MOZA not in r[3] and ok(r)]
    sel["対照 コマ枠(枠そのもの)"] = even(koma, un, rng, N_PER)
    beta = {(r[0], r[1]) for r in rows if r[3] == BETA and r[5] == "raster" and ok(r)}
    n_name = len(beta)
    com = {(r["page_path"], int(r["layer_id"])) for r in csv.DictReader(open(os.path.join(ROOT, TABLE), encoding="utf-8"))
           if BETA in r["author_comment"]}
    beta |= {k for k in com if ok(by[k])}
    sel["対照 髪や服のベタ"] = sorted(beta)
    print(f"    候補(.clip・恒久除外を外した後): 材料A {len(kuro)} / 材料B {len(moza):,} / 用紙 {len(paper):,} / コマ枠 {len(koma):,} / "
          f"ベタ 名前 {n_name} + 作者のコメント {len(com)}(重なりを畳んで {len(beta)})")
    print("    外した(全層を走査した延べ): " + " / ".join(f"{a} {b:,}" for a, b in drop.items()))
    seen, out = set(), []
    for s, ks in sel.items():
        dup = [k for k in ks if k in seen]
        ks = [k for k in ks if k not in seen]
        seen |= set(ks)
        print(f"    {s}: {len(ks)}層 / 頁 {len({k[0] for k in ks})} / 作品単位 {len({un[wd(k[0])] for k in ks})}"
              + (f"(先の系統と重なって飛ばした {len(dup)})" if dup else ""))
        for k in ks:
            out.append((s, k))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["no", "page_path", "layer_id", "group", "v_layer_kind", "is_folder"])
        for i, (s, k) in enumerate(out, 1):
            w.writerow([i, k[0], k[1], s, by[k][5], int(bool(by[k][4]))])
    print(f"    計 {len(out)}層 → `{OUT}`(非公開。発注は Q の追補の後)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
