#!/usr/bin/env python
"""**Q13 の画の清書の層が、コマ枠のフォルダーの中か外か**(Track Q の問い、2026-11-22。新しい走行ではない。DB の親子の記録だけ)。

- 層 = Q13 の便の層の点検表 `q13_synthetic_frames_layers_check_20261122.csv` の `role = line` の層のうち、
  `art_file` の在る頁(160)の層
- 中 = 構造 DB(10-26 の断面)の `parent_id` を根まで辿って、祖先に `v_layer_kind = frame_folder` が在る
  (そのフォルダーのマスクで切られる)。祖先のコマ枠フォルダーが Q13 の `frame` の行か、モザイクの名前だけか、どちらでもないかを分ける
- 外 = 祖先にコマ枠のフォルダーが無い(**はみ出しうる**であって、はみ出しているではない。コマの内側だけに描かれた外の層もある)
- 分からない = DB に層が無い / 親を辿れない(数えて印字する。規律9)
- マスクの有効・無効は DB の列に無い。`has_mask` だけを印字する

出力 `data/census_full/q13_line_layers_frame_side_20261122.csv`(層ごと)と `..._pages_20261122.csv`(頁ごと)。非公開。
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
DEL = os.path.join(ROOT, "..", "lineart-layer-exclusion", "data", "census")
PAGES = os.path.join(DEL, "q13_synthetic_frames_pages_20261122.csv")
CHECK = os.path.join(DEL, "q13_synthetic_frames_layers_check_20261122.csv")
OUT_L = os.path.join(D, "q13_line_layers_frame_side_20261122.csv")
OUT_P = os.path.join(D, "q13_line_layers_frame_side_pages_20261122.csv")
CLS = ["中(Q13 の frame のフォルダーの下)", "中(モザイクのコマ枠フォルダーだけ)", "中(frame の一覧に無いコマ枠フォルダー)",
       "外", "分からない(DB に層が無い)", "分からない(親を辿れない)"]


def main() -> int:
    pg = {r["page_path"]: r for r in csv.DictReader(open(PAGES, encoding="utf-8"))}
    art = {p for p, r in pg.items() if r["art_file"]}
    ck = list(csv.DictReader(open(CHECK, encoding="utf-8")))
    fr = {(r["page_path"], int(r["layer_id"])) for r in ck if r["role"] == "frame"}
    L = [r for r in ck if r["role"] == "line" and r["page_path"] in art]
    nl_other = sum(1 for r in ck if r["role"] == "line" and r["page_path"] not in art)
    print(f"art の在る頁 {len(art)} / その頁の清書の層 {len(L):,}(状態: " + " / ".join(
        f"{k} {v}" for k, v in collections.Counter(r["status"] for r in L).items()) + f")/ art の無い頁の清書の層 {nl_other}(数えない)")
    ps = sorted(art)
    c = db.connect()
    info = {(r[0], r[1]): r for r in c.execute(
        f"select l.page_path, l.layer_id, l.parent_id, coalesce(l.name,''), l.has_mask, v.v_layer_kind from layers l "
        f"join v_layer_kind v using (page_path, layer_id) where l.page_path in ({','.join('?' * len(ps))})", ps).fetchall()}
    out, per = [], collections.defaultdict(collections.Counter)
    for r in L:
        k = (r["page_path"], int(r["layer_id"]))
        cur, ffs, cls = info.get(k), [], None
        if cur is None:
            cls = CLS[4]
        else:
            seen, p = set(), cur[2]
            while p is not None:
                a = info.get((k[0], p))
                if a is None or p in seen:
                    cls = CLS[5]; break
                seen.add(p)
                if a[5] == "frame_folder":
                    ffs.append(a)
                p = a[2]
            if cls is None:
                if not ffs:
                    cls = CLS[3]
                elif any((k[0], a[1]) in fr for a in ffs):
                    cls = CLS[0]
                elif all("モザイク" in a[3] for a in ffs):
                    cls = CLS[1]
                else:
                    cls = CLS[2]
        per[k[0]][cls] += 1
        out.append(dict(q=r["q"], page_path=k[0], layer_id=k[1], side=cls,
                        frame_folder_ids=";".join(str(a[1]) for a in ffs),
                        n_px_alpha_gt0=r["n_px_alpha_gt0_masked"]))
    tot = collections.Counter(o["side"] for o in out)
    print("層: " + " / ".join(f"{k} {tot[k]:,}" for k in CLS))
    print(f"  祖先のコマ枠フォルダーの数(中の層): " + " / ".join(
        f"{k}つ {v}" for k, v in sorted(collections.Counter(len(o['frame_folder_ids'].split(';')) for o in out if o['frame_folder_ids']).items())))
    hm = collections.Counter(info[f][4] for f in fr if f in info and f[0] in art)
    print(f"  art の在る頁の frame のフォルダー {sum(hm.values())}: has_mask " + " / ".join(f"{k} {v}" for k, v in hm.items()))
    px = collections.Counter()
    for o in out:
        px[o["side"]] += int(o["n_px_alpha_gt0"] or 0)
    print("  画素(層自身のマスクの後のアルファ>0 の和、重なりを重ねて数える): " + " / ".join(f"{k} {px[k]:,}" for k in CLS if px[k]))
    prow = []
    for p in ps:
        x = per[p]
        prow.append(dict(q=pg[p]["q"], split=pg[p]["split"], page_path=p, n_in=x[CLS[0]] + x[CLS[1]] + x[CLS[2]],
                         n_out=x[CLS[3]], n_unknown=x[CLS[4]] + x[CLS[5]]))
    a_out = sum(1 for r in prow if r["n_in"] == 0 and r["n_out"] > 0)
    a_in = sum(1 for r in prow if r["n_out"] == 0)
    print(f"頁 {len(prow)}: 全部外 {a_out} / 全部中 {a_in} / 混ざる {len(prow) - a_out - a_in}")
    for s in ("tune", "check"):
        rs = [r for r in prow if r["split"] == s]
        print(f"  {s}: 頁 {len(rs)} / 外の層が1つ以上 {sum(r['n_out'] > 0 for r in rs)} / 中 {sum(r['n_in'] for r in rs):,} 層 / 外 {sum(r['n_out'] for r in rs):,} 層")
    for f, rs in ((OUT_L, out), (OUT_P, prow)):
        with open(f, "w", encoding="utf-8", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rs[0])); wr.writeheader(); wr.writerows(rs)
        print(f"出力: {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
