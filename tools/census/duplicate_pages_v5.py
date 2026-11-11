#!/usr/bin/env python
"""**対 v5 の中で二重に入っている頁の一覧** —— 基盤の回送(`reply_census_duplicate_pages_20261112.md`、ユーザーの決定
「二重に入ってるページのどれを使うかはcensus」)と `reply_census_pairs_v5_arrived_20261112.md`。

**決めたこと(数える前に)**:
- 母集団: 対 v5 の QC(`clip_pairs_qc_v5_20261112.csv`、1,848行)。**`pair_quality` が `ok` でない行を先に落とす**(理由別に数える)。
  恒久除外の頁も落とす
- **二重の定義**(D の聞き取りの言い方「同じ作品・同じ頁番号が別フォルダーから」): 鍵 = (年の札 \\ 作品名 = パスの頭2つ, ファイル名)。
  同じ鍵に**別のフォルダー**の頁が2つ以上あれば一組。**中身が同じかは見ない**(QC の `duplicate_group` は中身の指紋の重複で、別の問い)
- **どれを採るかは決めない**(版の同定は作者に聞く。2026-10-26)。この器具は組を並べるだけ。組ごとに構造 DB の事実
  (層の数・キャンバス・形式)を並べる。名前や統計量で採る側を推さない

    ./venv/bin/python tools/census/duplicate_pages_v5.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QC = "inbox/clip_pairs_qc_v5_20261112.csv"
QC_SHA = None   # 初回の値を印字して固定する
OUT = "data/census_full/duplicate_pages_v5_20261112.csv"


def key(p):
    parts = p.split("\\")
    # F: \ manuscripts \ <年> \ <作品> \ ... \ <ファイル>
    return (parts[2] + "\\" + parts[3], parts[-1]) if len(parts) >= 5 else None


def main() -> int:
    print("[対 v5 の中で二重に入っている頁]")
    sha = hashlib.sha256(open(os.path.join(ROOT, QC), "rb").read()).hexdigest()
    print(f"    QC `{QC}` sha256 {sha[:16]}")
    ex = exclusions.load()
    rows = list(csv.DictReader(open(os.path.join(ROOT, QC), encoding="utf-8")))
    drop = collections.Counter()
    keep = []
    for r in rows:
        if r["pair_quality"] != "ok":
            drop[f"QC {r['pair_quality']}"] += 1
        elif r["source_path"] in ex:
            drop["恒久除外の頁"] += 1
        else:
            keep.append(r)
    print(f"    QC {len(rows):,}行 → 落とした " + " / ".join(f"{a} {b}" for a, b in drop.most_common()) + f" → {len(keep):,}行")
    pages = sorted({r["source_path"] for r in keep})
    print(f"    頁 {len(pages):,}(1頁に対が複数ある行を畳んだ数)")
    by = collections.defaultdict(set)
    nokey = 0
    for p in pages:
        k = key(p)
        if k is None:
            nokey += 1
            continue
        by[k].add(p)
    groups = {k: sorted(v) for k, v in by.items() if len({p.rsplit(chr(92), 1)[0] for p in v}) >= 2}
    print(f"    鍵の作れない頁 {nokey} / **二重の組 {len(groups)}**(頁 {sum(len(v) for v in groups.values())})")
    sz = collections.Counter(len(v) for v in groups.values())
    print("    組の大きさ: " + " / ".join(f"{a}頁 {b}組" for a, b in sorted(sz.items())))
    con = db.connect()
    allp = sorted({p for v in groups.values() for p in v})
    q = ",".join("?" * len(allp))
    info = {r[0]: r[1:] for r in con.execute(f"""select p.page_path, p.format, p.v_canvas_px_w, p.v_canvas_px_h, count(l.layer_id)
                                                     from pages p left join layers l using (page_path)
                                                    where p.page_path in ({q}) group by 1, 2, 3, 4""", allp).fetchall()}
    qc_by = collections.defaultdict(list)
    for r in keep:
        qc_by[r["source_path"]].append(r)
    out = []
    for gi, (k, ps) in enumerate(sorted(groups.items()), 1):
        for p in ps:
            f, w, h, n = info.get(p, ("", "", "", ""))
            out.append(dict(group=gi, work=k[0], file=k[1], page_path=p, folder=p.rsplit(chr(92), 1)[0], format=f,
                            canvas=f"{w}x{h}", n_layers=n, n_pairs_v5=len(qc_by[p]),
                            content_fingerprints=" ".join(sorted({x["content_fingerprint"] for x in qc_by[p]}))))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    nw = collections.Counter(d["work"] for d in out if True)
    print(f"    作品(年\\作品名) {len(nw)}: " + " / ".join(f"{a} {b}頁" for a, b in nw.most_common(8)) + (" …" if len(nw) > 8 else ""))
    same_fp = sum(1 for k, ps in groups.items()
                  if len({frozenset(x["content_fingerprint"] for x in qc_by[p]) for p in ps}) == 1)
    print(f"    参考: 組の中で対の指紋がすべて同じ組 {same_fp} / {len(groups)}(中身が同じ。採る側の判断には使わない)")
    print(f"    → `{OUT}`(非公開)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
