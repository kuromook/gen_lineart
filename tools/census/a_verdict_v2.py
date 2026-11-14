#!/usr/bin/env python
"""**a の層ごとの判定 v2** —— census が E の規則を当てて作り直し、R・Q に配る(E の頼み、2026-11-14)。

**数を見る前に決めたこと**(規則は E の表と `empty_clause_apply.py` のもの。census は規則を作らない):
- 入力: `a_verdict_20261112.csv`(名前・種別・列。11-12)/ 全頁の画素のまとめた表(`empty_c` / `has_image_material` /
  `self_shown` / `ancestors_shown` / `pixel_formats`)
- 当てる順:
  1. 11-12 に `template_excluded` の層 → 理由「名前・種別・列」
  2. 空=1 で、画像素材でも形式5 でもない層 → 「条項:空(定義C)」(作者の裁定 11-13: 名前が『除外しない』の層も空なら落ちる)
  3. 画像素材の層(`has_image_material` = 1)→ 「条項:image_material(見えない / 出力に画素が0 / 出力に画素あり / 測れない)」
     (作者の指示 11-14 で 確定。理由を層ごとに名乗らせ、撤回を理由単位でできるようにする)
  4. 空=1 だが形式5 を含む層 → 落とさず「空だが保留(形式5 を含む(調査中。分析対象外))」
- **`not_excluded` の層にも、当たった『除外しない』の行の (名前, 種別) を `a_rule_v2` に書く**(E の頼み5。落ちた数を名前別に印字できるように)
- **E の層ごとの判定**(`../lineart-aesthetic-judge/data/role_predictor/a_verdict_with_empty_20261114.csv`)**と1行ずつ突き合わせ**、
  状態の不一致を数で印字する(理由の文字の違いは別に数える。`not_excluded` の行に名前を足した分は違って当然なので除いて比べる)

    ./venv/bin/python tools/census/a_verdict_v2.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
from a_open_items_20261112 import E_TABLE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = "data/census_full/a_verdict_20261112.csv"
V1_SHA = "0fba46450155b0a6"
PIX = "data/census_full/full_clip_pixels_layers_merged.csv"
PIX_SHA = "d3d1367808f5cb1a"
E_V2 = os.path.join(os.path.dirname(ROOT), "lineart-aesthetic-judge", "data", "role_predictor", "a_verdict_with_empty_20261114.csv")
OUT = "data/census_full/a_verdict_v2_20261114.csv"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> int:
    print("[a の層ごとの判定 v2(E の規則を census が当てる)]")
    for p, s in ((V1, V1_SHA), (PIX, PIX_SHA)):
        if not sha(os.path.join(ROOT, p)).startswith(s):
            print(f"    **sha256 が違う** {p}")
            return 1
    print(f"    E の表 sha256 {sha(E_TABLE)[:16]}(読むだけ。11-12 の判定の『除外しない』は a_verdict_20261112 の側に入っている)")
    pix = {(r["page_path"], r["layer_id"]): r for r in csv.DictReader(open(os.path.join(ROOT, PIX), encoding="utf-8"))}
    con = db.connect()
    nk = {(p, str(i)): (n or "", k) for p, i, n, k in con.execute(
        "select l.page_path, l.layer_id, l.name, v.v_layer_kind from layers l join v_layer_kind v using (page_path, layer_id)").fetchall()}
    out, st = [], collections.Counter()
    for r in csv.DictReader(open(os.path.join(ROOT, V1), encoding="utf-8")):
        k = (r["page_path"], r["layer_id"])
        x = pix.get(k)
        e = x["empty_c"] if x else ""
        im = bool(x) and x["has_image_material"] == "1"
        f5 = bool(x) and "5" in x["pixel_formats"].split(";")
        if r["a_verdict"] == "template_excluded":
            v, why = "template_excluded", "名前・種別・列"
        elif e == "1" and not im and not f5:
            v, why = "template_excluded", "条項:空(定義C)"
        elif im:
            vis = x["self_shown"] == "1" and x["ancestors_shown"] == "1"
            s = "見えない" if not vis else "出力に画素が0" if e == "1" else "出力に画素あり" if e == "0" else "測れない"
            v, why = "template_excluded", f"条項:image_material({s})"
        elif e == "1" and f5:
            v, why = r["a_verdict"], "空だが保留(形式5 を含む(調査中。分析対象外))"
        else:
            v, why = r["a_verdict"], ""
        name = ""
        if r["a_verdict"] == "not_excluded":
            name = "除外しない: {} ({})".format(*nk[k])
        st[(v, why.split("(")[0] if why else "")] += 1
        out.append(dict(page_path=k[0], layer_id=k[1], a_verdict_v2=v, a_rule_v2=why, keep_name=name))
    n = len(out)
    print(f"    母集団: {n:,}層(構造 DB の全層から恒久除外の頁の層を外したもの)")
    for (v, w), c in sorted(st.items(), key=lambda z: -z[1]):
        print(f"        {v} / {w or '—'}: {c:,}")
    tx = sum(c for (v, _), c in st.items() if v == "template_excluded")
    print(f"    落とす 計 {tx:,} / {n:,} = {tx / n:.4f}(前 11-12: 196,291 / {n:,} = {196291 / n:.4f})")
    sub = collections.Counter(r["a_rule_v2"] for r in out if r["a_rule_v2"].startswith("条項:image_material"))
    print("    image_material の内訳: " + " / ".join(f"{k.split('(')[1][:-1]} {c:,}" for k, c in sub.most_common()))
    kn = collections.Counter(r["keep_name"] for r in out if r["keep_name"] and r["a_rule_v2"].startswith("条項:空"))
    print(f"    名前が『除外しない』の層で空として落ちた層 {sum(kn.values()):,}(名前別、上位 8): "
          + " / ".join(f"{k[6:]} {c:,}" for k, c in kn.most_common(8)))
    # E と突き合わせ
    if os.path.exists(E_V2):
        ev = {(r["page_path"], r["layer_id"]): r for r in csv.DictReader(open(E_V2, encoding="utf-8"))}
        dv = sum(1 for r in out if ev[(r["page_path"], r["layer_id"])]["a_verdict_v2"] != r["a_verdict_v2"])
        dr = sum(1 for r in out if ev[(r["page_path"], r["layer_id"])]["a_rule_v2"] != r["a_rule_v2"])
        print(f"    E の層ごとの判定(sha256 {sha(E_V2)[:16]}、{len(ev):,}行)と突き合わせ: 状態の不一致 {dv} / 理由の文字の不一致 {dr}")
    else:
        print("    E の層ごとの判定が見つからない(突き合わせていない)")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"    → `{OUT}`(非公開、{n:,}行)。E の表のパス {os.path.relpath(E_TABLE, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
