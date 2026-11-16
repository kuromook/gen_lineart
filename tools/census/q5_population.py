#!/usr/bin/env python
"""**3D背景 Q5 の母集団の一覧と、層だけの画の発注書**(Track Q の頼み 11-16 22:55、ユーザーの承認「ア」「出してよい」)。

**数を見る前に決めたこと**(census は一覧を引くだけ。点数は無い。下限の規則は Q の登録):
- 出発: `image_material_structure_20261114.csv`(構造 DB の `has_image_material` の層、恒久除外の頁を外した後)
- 落とす理由(先に当たった1つで数える。規律9):
  1. 種別(`v_layer_kind`)が image_material でない(text・filter など先に当たる種別)
  2. CSP で見えない(`layer_visibility_20261116.csv` の `shown_in_csp` が 1 でない。層か祖先が非表示)
  3. 表示の点検が調査中(`vis_check` に `調査中` を含む。**.lip の `画素の表に無い` は調査中ではない** ——
     初版は `一致` でないものを落としていて .lip 774層を誤って落とした。送る前に直した。前後の数は印字に残す)
  4. E の二択2便の100層(`image_material_check100_names_20261116.csv`)
  5. sha256 の束の表に無い頁 / 6. 代表でない頁(`page_representative.csv`)
- `.lip` は落とさず一覧に残す(Q の条件に形式が無い)。**但し `.lip` に画素は無いので、発注書は `.clip` だけ**。数を分けて印字する
- 恒久除外の門は `exclusions` で引き直して 0 を確かめる
- 列: page_path / layer_id / format / layer_name / ancestors / split_unit / year_label(頁の札)

    ./venv/bin/python tools/census/q5_population.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ST = "data/census_full/image_material_structure_20261114.csv"
VIS = "data/census_full/layer_visibility_20261116.csv"
VIS_SHA = "686274775aff7398"
E100 = "data/census_full/image_material_check100_names_20261116.csv"
E100_SHA = "e98ad5771f901418"
REP = "data/census_full/page_representative.csv"
OUT = "data/census_full/q5_population_20261116.csv"
ORDER = "outbox/q5_layer_images_order_20261116.csv"


def rd(p):
    return list(csv.DictReader(open(os.path.join(ROOT, p), encoding="utf-8")))


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def main() -> int:
    print("[3D背景 Q5 の母集団]")
    for p, s in ((VIS, VIS_SHA), (E100, E100_SHA)):
        if not sha(p).startswith(s):
            print(f"    **sha256 が違う** {p}")
            return 1
    ex = exclusions.load()
    un = split_units.units(db.connect())
    st = rd(ST)
    vis = {(r["page_path"], r["layer_id"]): r for r in rd(VIS)}
    e100 = {(r["page_path"], r["layer_id"]) for r in rd(E100)}
    rep = {r["page_path"]: r for r in rd(REP)}
    print(f"    出発: has_image_material の層(恒久除外の頁を外した後){len(st):,}"
          f"(.clip {sum(r['format'] == 'clip' for r in st):,} / .lip {sum(r['format'] == 'lip' for r in st):,})"
          f" / E の100層のうち出発に在る {len(e100 & {(r['page_path'], r['layer_id']) for r in st})}")
    drop, out = collections.Counter(), []
    for r in st:
        k = (r["page_path"], r["layer_id"])
        v = vis.get(k)
        for why, bad in (("恒久除外の頁", k[0] in ex),
                         ("種別が image_material でない", r["v_layer_kind"] != "image_material"),
                         ("表示の表に無い", v is None),
                         ("CSP で見えない(層か祖先が非表示)", v is not None and v["shown_in_csp"] != "1"),
                         ("表示の点検が調査中", v is not None and "調査中" in v["vis_check"]),
                         ("E の二択2便の100層", k in e100),
                         ("sha256 の束の表に無い頁", k[0] not in rep),
                         ("代表でない頁", k[0] in rep and rep[k[0]]["is_representative"] != "yes")):
            if bad:
                drop[(r["format"], why)] += 1
                break
        else:
            out.append(dict(page_path=k[0], layer_id=k[1], format=r["format"], layer_name=r["name"],
                            ancestors=r["ancestors"], split_unit=un[wd(k[0])], year_label=rep[k[0]]["year_label"]))
    for f in ("clip", "lip"):
        print(f"    落とした .{f}(先に当たった理由1つ): "
              + (" / ".join(f"{w} {n:,}" for (ff, w), n in drop.items() if ff == f) or "なし"))
    stk = {(r["page_path"], r["layer_id"]) for r in st}
    vc = collections.Counter(vis[k]["vis_check"] for k in stk if k in vis)
    print("    (参考)出発の層の vis_check: " + " / ".join(f"{a} {b:,}" for a, b in vc.most_common()))
    clip = [x for x in out if x["format"] == "clip"]
    print(f"    残った層 {len(out):,}(.clip {len(clip):,} / .lip {len(out) - len(clip):,})"
          f" / 頁 {len({x['page_path'] for x in out}):,} / 作品単位 {len({x['split_unit'] for x in out})}")
    print("    札の内訳(.clip): " + " / ".join(f"{a} {b}" for a, b in sorted(collections.Counter(x['year_label'] for x in clip).items())))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["no"] + list(out[0]))
        w.writeheader()
        for i, x in enumerate(out, 1):
            w.writerow(dict(no=i, **x))
    with open(os.path.join(ROOT, ORDER), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pos", "page_path", "layer_id"])
        for i, x in enumerate(clip, 1):
            w.writerow([i, x["page_path"], x["layer_id"]])
    print(f"    → `{OUT}`(非公開、{len(out):,}行。名前を含む)/ 発注書 `{ORDER}`(.clip {len(clip):,}層、名前なし)")
    print(f"    sha256: 一覧 {sha(OUT)[:16]} / 発注書 {sha(ORDER)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
