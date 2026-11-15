#!/usr/bin/env python
"""**a の層ごとの判定 v3** —— 画像素材の50枚の二択の棒で、理由『image_material(出力に画素あり)』を外す(E の頼み、2026-11-15)。

**数を見る前に決めたこと**(規則と棒は E の登録 `image_material_50_image_check_20261114`。census は規則を作らない):
- 作者の答え: E の `image_material_check50_answers_v2_20261115.jsonl`(追記だけ)。**層ごとに最後の行**を採る(E と同じ)。
  第1版の11行(壊れた画)は数えない(版として写しだけ残す)
- 素の内訳を印字し、E の数(資料 42 / 絵の一部 8)と突き合わせる。1行目と27行目(バイトが同じファイルの同じ層)の一致を印字する
- 棒は E のもの: 絵の一部が 3〜9 → 理由『条項:image_material(出力に画素あり)』だけを外して候補に戻す。
  **census はこの棒を読み直さない**(E の判定を当てるだけ)
- 外した層の状態は **11-12 の判定(`a_verdict_20261112.csv`)の状態に戻す**。v2 の理由は `a_rule_v2` に残し、
  `a_rule_v3` に「外した:…」と書く(撤回を理由単位でできるように)
- **但し書き**: 『出力に画素あり』の分け方は全頁の画素の表から作ったもので、画像素材の層では古い切れ端を数えている
  (`census/caveats.csv` の T129 の行)。**外した 821層は「配置された画像が頁に出ている層」ではない**
- 前の数(v2)と後の数(v3)を両方、名前を付けて印字する(規律10)

    ./venv/bin/python tools/census/a_verdict_v3.py
"""
import collections
import csv
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = "data/census_full/a_verdict_20261112.csv"
V2 = "data/census_full/a_verdict_v2_20261114.csv"
V2_SHA = "daff0879fe3543cb"
ANS = "data/census_full/image_material_check50_answers_v2_20261115.jsonl"
ANS_SHA = "5b5dd8c969630a24"
ORDER = "outbox/image_material_check50_images_order_20261114.csv"
OUT = "data/census_full/a_verdict_v3_20261115.csv"
OUT_ANS = "data/census_full/image_material_check50_answers_v2_last_20261115.csv"
DROP = "条項:image_material(出力に画素あり)"


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def main() -> int:
    print("[a の層ごとの判定 v3(50枚の二択の棒を当てる)]")
    for p, s in ((V2, V2_SHA), (ANS, ANS_SHA)):
        if not sha(p).startswith(s):
            print(f"    **sha256 が違う** {p}")
            return 1
    lines = [json.loads(x) for x in open(os.path.join(ROOT, ANS), encoding="utf-8") if x.strip()]
    last = {}
    for x in lines:
        last[x["item_id"]] = x
    changed = sum(1 for k in last if len({x["label"] for x in lines if x["item_id"] == k}) > 1)
    print(f"    答えの行 {len(lines)} → 層ごとの最後の行 {len(last)}(答えを直した層 {sum(1 for k in last if sum(x['item_id'] == k for x in lines) > 1)}、うちラベルが変わった層 {changed})")
    c = collections.Counter(x["label"] for x in last.values())
    print("    素の内訳: " + " / ".join(f"{k} {v}" for k, v in c.most_common()) + "(E の数: 資料 42 / 絵の一部 8)")
    order = {f"im50_{int(r['pos']):02d}": r for r in csv.DictReader(open(os.path.join(ROOT, ORDER), encoding="utf-8"))}
    print(f"    1行目と27行目(バイトが同じファイルの同じ層): {last['im50_01']['label']} / {last['im50_27']['label']}")
    with open(os.path.join(ROOT, OUT_ANS), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pos", "page_path", "layer_id", "label", "value", "comment"])
        for k in sorted(last):
            w.writerow([int(k[5:]), order[k]["page_path"], order[k]["layer_id"], last[k]["label"], last[k]["value"], last[k]["comment"]])
    v1 = {(r["page_path"], r["layer_id"]): r["a_verdict"] for r in csv.DictReader(open(os.path.join(ROOT, V1), encoding="utf-8"))}
    out, before, after = [], collections.Counter(), collections.Counter()
    for r in csv.DictReader(open(os.path.join(ROOT, V2), encoding="utf-8")):
        k = (r["page_path"], r["layer_id"])
        v, why = r["a_verdict_v2"], r["a_rule_v2"]
        before[v] += 1
        if why == DROP:
            v, why3 = v1[k], f"外した:{DROP[3:]}(50枚の二択 11-15。絵の一部 8/50 が棒 3〜9 に入った。E の規則)"
        else:
            why3 = why
        after[v] += 1
        out.append(dict(page_path=k[0], layer_id=k[1], a_verdict_v3=v, a_rule_v3=why3, a_rule_v2=r["a_rule_v2"], keep_name=r["keep_name"]))
    n = len(out)
    b, a = before["template_excluded"], after["template_excluded"]
    print(f"    落とす: v2(11-14){b:,} / {n:,} = {b / n:.4f} → v3(11-15){a:,} / {n:,} = {a / n:.4f}(戻した {b - a:,})")
    back = collections.Counter(x["a_verdict_v3"] for x in out if x["a_rule_v3"].startswith("外した"))
    print("    戻した層の状態(11-12 の判定): " + " / ".join(f"{k} {v:,}" for k, v in back.most_common()))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"    → `{OUT}`(非公開、{n:,}行)/ 答えの最後の行 → `{OUT_ANS}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
