#!/usr/bin/env python
"""**Track Q の黒消し v3 の17行の聞き取り: 並べ順の鍵と、画の一覧** —— Q の依頼
(`note_q_census_q3_trial17_approved_20261116.md`、ユーザーの承認「1外す　2OK」2026-11-16)。
Q1 の器具(`hearing_q1_trial17.py`)を、入力・種・名前だけ替えて写した。並べ方は同じ。
**抽出側は census の発注の便より先に、同じやり方で並べた一覧(`config/q3_trial17/order.csv`)で画を描き始めていた**(11-16 19:07)。
この器具の一覧と1行ずつ突き合わせる(印字する)。
**表示**: 層と祖先の表示(`visibility & 1`、`layer_role_table.page_order`)を鍵に足し、CSP で見えない層の行を印字する(Q の頼み2)。

**決めたこと(Q の依頼のまま。行は足さない・差し替えない)**:
- 入力: Q の一覧 `q3_trial17_layers_20261116.csv`(17行)。sha256 を確かめる。恒久除外の頁が入っていれば止める
- 並べ順: `random.Random(20261112)` で混ぜ、出した位置(`pos`)を鍵に書く
- **作者に見せる表には `cell` / `group` / `era` / `A2` / `A2m` / `A3c` / `A3r` / `A4` を出さない**(Q)。**層の名前も出さない**(census の判断:
  名前が当て方の片方なので、名前を見せると名前の当て方に寄せる)。画のファイル名も位置だけで付ける
- 画は行ごとに2枚(頁の合成 / 問う層だけ、白の背景)。原寸。census は画を開かない(原稿のラスタを読まない)。届いたら Q へ写すだけ

    ./venv/bin/python tools/census/hearing_q1_trial17.py
"""
import csv
import hashlib
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
from layer_role_table import page_order  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = "inbox/q3_trial17_layers_20261116.csv"
SRC_SHA = "39e8457f6552ef79"
SEED = 20261116
KEY_OUT = "data/census_full/hearing_q3_trial17_key_20261116.csv"
SHOW_OUT = "data/census_full/hearing_q3_trial17_show_20261116.csv"
ORDER_OUT = "outbox/q3_trial17_images_order_20261116.csv"


def main() -> int:
    print(f"[黒消しの17行の聞き取り(Track Q)] seed {SEED}")
    sha = hashlib.sha256(open(os.path.join(ROOT, SRC), "rb").read()).hexdigest()
    if not sha.startswith(SRC_SHA):
        print(f"    **sha256 が違う** {sha[:16]}")
        return 1
    rows = list(csv.DictReader(open(os.path.join(ROOT, SRC), encoding="utf-8")))
    ex = exclusions.load()
    bad = [r for r in rows if r["page_path"] in ex]
    if bad:
        print(f"    **恒久除外の頁が {len(bad)} 行ある** —— 止める")
        return 1
    con = db.connect()
    po = page_order(con, sorted({r["page_path"] for r in rows}))
    vis = {}
    for r in rows:
        o = po.get((r["page_path"], int(r["layer_id"])))
        me = con.execute("select visibility & 1 from layers where page_path = ? and layer_id = ?", [r["page_path"], int(r["layer_id"])]).fetchone()
        vis[(r["page_path"], r["layer_id"])] = (int(me[0]) if me else "", o[3] if o else "")
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    key, show, img = [], [], []
    for pos, i in enumerate(order, 1):
        r = rows[i]
        a, b = f"q3t17_{pos:02d}_page.png", f"q3t17_{pos:02d}_layer.png"
        sv, av = vis[(r["page_path"], r["layer_id"])]
        key.append(dict(pos=pos, src_row=i + 1, **r, image_page=a, image_layer=b, self_shown_db=sv, ancestors_shown_db=av))
        show.append(dict(pos=pos, image_page=a, image_layer=b))
        img.append([pos, r["page_path"], r["layer_id"], a, b])
    for path, data in ((KEY_OUT, key), (SHOW_OUT, show)):
        with open(os.path.join(ROOT, path), "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    with open(os.path.join(ROOT, ORDER_OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pos", "page_path", "layer_id", "image_page", "image_layer"])
        w.writerows(img)
    hid = [k for k in key if not (k["self_shown_db"] == 1 and k["ancestors_shown_db"] == 1)]
    print(f"    CSP で見えない層(層か祖先が非表示): {len(hid)}行" + "".join(f" / pos {k['pos']}(層 {k['self_shown_db']}・祖先 {k['ancestors_shown_db']})" for k in hid))
    print(f"    {len(rows)}行(sha256 {sha[:16]})→ 鍵 `{KEY_OUT}` / 見せる表 `{SHOW_OUT}`(位置と画のファイル名だけ)/ 発注の一覧 `{ORDER_OUT}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
