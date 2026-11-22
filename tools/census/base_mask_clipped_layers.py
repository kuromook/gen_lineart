#!/usr/bin/env python
"""**背景のベースの層に、何がクリッピングされているか**(作者の言明 2026-11-23 の文脈を数で並べる。DB の引き直し。新しい走行ではない)。

作者(逐語、2026-11-23): 「前回の問では、クリッピングされた背景が乗るので、ベースマスクを削除してはいけないと答えた　原稿から背景がきえるからだ
　３D背景は「絵ではない」扱いとするなら消えて問題はない　しかし、ここは文脈を整理したほうがいい　単に「消していい」という言葉だけで処理すると矛盾がおきる」

- 層 = 名前がちょうど `bg base` / `base` / `rendering_base_back0001`(作者が名を挙げた3つ)の、フォルダーでない層
- クリッピングされている = 同じ親の中で、その層のすぐ上(`order_in_parent` が大きい側。`z_global` で向きを確かめた)から続く `clip_raw = 1` の兄弟。
  兄弟がフォルダーなら、その中の層も全部数える(フォルダーごと切られる)
- 乗っている層を分ける: 3D(`v_layer_kind = 3d` か、名前が `rendering_` で始まる)/ それ以外(手描きの背景・モブなどを含む。中身は名前では決めない)/ フォルダー
- 落とした・分からないものを数える(規律9)

出力 `data/census_full/base_mask_clipped_layers_20261123.csv`(ベースの層ごと。非公開)。
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "census_full", "base_mask_clipped_layers_20261123.csv")
NAMES = ["bg base", "base", "rendering_base_back0001"]


def main() -> int:
    c = db.connect()
    bases = c.execute(f"select page_path, layer_id from layers where name in ({','.join('?' * len(NAMES))}) and not folder", NAMES).fetchall()
    pages = sorted({p for p, _ in bases})
    L = {}
    for i in range(0, len(pages), 900):
        ch = pages[i:i + 900]
        for r in c.execute(f"select l.page_path, l.layer_id, l.parent_id, l.order_in_parent, l.z_global, l.clip_raw, l.folder, "
                           f"coalesce(l.name,''), v.v_layer_kind from layers l join v_layer_kind v using (page_path, layer_id) "
                           f"where l.page_path in ({','.join('?' * len(ch))})", ch).fetchall():
            L[(r[0], r[1])] = r
    kids = collections.defaultdict(list)
    for r in L.values():
        kids[(r[0], r[2])].append(r)
    for v in kids.values():
        v.sort(key=lambda r: (r[3] is None, r[3] if r[3] is not None else 0, r[4] or 0))
    bad_dir = sum(1 for v in kids.values() for a, b in zip(v, v[1:]) if None not in (a[3], b[3], a[4], b[4]) and a[4] > b[4])
    no_order = sum(1 for p, i in bases if L[(p, i)][3] is None)
    print(f"order_in_parent が空のベースの層 {no_order}")
    print(f"ベースの層 {len(bases):,}(頁 {len(pages):,})/ 兄弟の並びで order と z が逆向きの組 {bad_dir}")

    def desc(r):
        out = []
        for k in kids.get((r[0], r[1]), []):
            out.append(k)
            if k[6]:
                out += desc(k)
        return out

    rows, tot = [], collections.Counter()
    for p, i in bases:
        b = L[(p, i)]
        sib = kids[(p, b[2])]
        j = sib.index(b)
        clipped = []
        for s in sib[j + 1:]:
            if s[5] != 1:
                break
            clipped.append(s)
        allc = []
        for s in clipped:
            allc.append(s)
            if s[6]:
                allc += desc(s)
        n3d = sum(1 for s in allc if not s[6] and (s[8] == "3d" or s[7].startswith("rendering_")))
        nfo = sum(1 for s in allc if s[6])
        noth = sum(1 for s in allc if not s[6]) - n3d
        rows.append(dict(page_path=p, layer_id=i, base_name=b[7], n_clipped_direct=len(clipped),
                         n_clipped_layers_3d=n3d, n_clipped_layers_other=noth, n_clipped_folders=nfo))
        tot[(b[7], "has" if clipped else "none")] += 1
    for n in NAMES:
        rs = [r for r in rows if r["base_name"] == n]
        has = [r for r in rs if r["n_clipped_direct"]]
        print(f"  {n}: 層 {len(rs):,} / 何かがクリッピングされている {len(has):,} / 何も無い {len(rs) - len(has):,}")
        print(f"     乗っている層の計: 3D {sum(r['n_clipped_layers_3d'] for r in rs):,} / それ以外 {sum(r['n_clipped_layers_other'] for r in rs):,}"
              f" / フォルダー {sum(r['n_clipped_folders'] for r in rs):,}")
        print(f"     ベースの層ごと: 3D だけ {sum(1 for r in has if r['n_clipped_layers_3d'] and not r['n_clipped_layers_other'])}"
              f" / それ以外だけ {sum(1 for r in has if r['n_clipped_layers_other'] and not r['n_clipped_layers_3d'])}"
              f" / 両方 {sum(1 for r in has if r['n_clipped_layers_3d'] and r['n_clipped_layers_other'])}"
              f" / フォルダーだけ(中に層なし) {sum(1 for r in has if not r['n_clipped_layers_3d'] and not r['n_clipped_layers_other'])}")
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"出力: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
