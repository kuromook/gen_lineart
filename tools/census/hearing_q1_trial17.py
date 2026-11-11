#!/usr/bin/env python
"""**Track Q の黒消しの17行の聞き取り: 並べ順の鍵と、抽出側への画の発注の一覧** —— Q の依頼
(`note_q_census_q1_trial17_approved_20261112.md`、ユーザーの承認「承認」2026-11-12)。

**決めたこと(Q の依頼のまま。行は足さない・差し替えない)**:
- 入力: Q の一覧 `q1_trial17_layers_20261112.csv`(17行)。sha256 を確かめる。恒久除外の頁が入っていれば止める
- 並べ順: `random.Random(20261112)` で混ぜ、出した位置(`pos`)を鍵に書く
- **作者に見せる表には `cell` / `group` / `A2` / `A3` / `A4` を出さない**(Q)。**層の名前も出さない**(census の判断:
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
import exclusions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = "inbox/q1_trial17_layers_20261112.csv"
SRC_SHA = "55872305a5795c11"
SEED = 20261112
KEY_OUT = "data/census_full/hearing_q1_trial17_key_20261112.csv"
SHOW_OUT = "data/census_full/hearing_q1_trial17_show_20261112.csv"
ORDER_OUT = "outbox/q1_trial17_images_order_20261112.csv"


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
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    key, show, img = [], [], []
    for pos, i in enumerate(order, 1):
        r = rows[i]
        a, b = f"q1t17_{pos:02d}_page.png", f"q1t17_{pos:02d}_layer.png"
        key.append(dict(pos=pos, src_row=i + 1, **r, image_page=a, image_layer=b))
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
    print(f"    {len(rows)}行(sha256 {sha[:16]})→ 鍵 `{KEY_OUT}` / 見せる表 `{SHOW_OUT}`(位置と画のファイル名だけ)/ 発注の一覧 `{ORDER_OUT}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
