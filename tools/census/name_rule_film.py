#!/usr/bin/env python
"""**名前の決まり + 膜の決まり F**(`name_rule_film` v1)を、`.clip` の全層に当てる(ユーザーの決定、2026-11-20、Track R 経由)。

ユーザー(R4 の結果を見て):「悪くなさそうだし今censusのフローに入れてもいいと思う」。登録は R の木
`config/prereg_r4_film_rule_nashi_20261120.md`(a2c4b00)。**census は決まりを作らない。凍らせた形のまま当てる**:
- 対象は `name_rule` = なし の層だけ。膜 = `n_px_a_ge26 / n_px_a_gt0 < 0.5`
- F: なし かつ 膜 → `あたり+下絵(膜)`。名前の決まりが言う層は上書きしない。膜でない層・画素の無い層には何も言わない

**率を添える(規律13)**。census は R の答えのファイルと census の組の鍵で引き直して印字する
(R の数を写さない)。的中は**確かめた母集団(R4 の門の後)の中だけ**の数で、門の外(学習の作品・E の除外)の層には
列を付けるが、`in_r4_population` = 0 と書く。**取りこぼしは測っていない**。

出力 `data/census_full/layer_name_rule_film_v1_20261120.csv`(頁のパスを持つので公開しない・追跡しない)。
"""
import collections
import csv
import json
import os
import sys

from scipy.stats import beta, fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
from stage_hearing import name_rule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
PIX = os.path.join(D, "full_clip_pixels_layers_merged.csv")
POP = os.path.join(D, "r4_film_nashi_population_20261120.csv")
KEY = os.path.join(D, "r4_film_nashi_hearing_key_20261120.csv")
ANS = os.path.join(ROOT, "..", "lineart-layer-picture", "data", "census",
                   "r4_film_nashi_images_20261120", "r4_answers.jsonl")
OUT = os.path.join(D, "layer_name_rule_film_v1_20261120.csv")
MAP = {"atari": "あたり+下絵", "shitae": "あたり+下絵", "line": "清書", "__none__": "段ではない"}
F_LABEL = "あたり+下絵(膜)"


def rescore():
    last = {}
    for line in open(ANS, encoding="utf-8"):
        r = json.loads(line)
        last[r["item_id"]] = r
    key = {r["row"]: r["group"] for r in csv.DictReader(open(KEY, encoding="utf-8"))}
    assert len(last) == len(key) == 30, (len(last), len(key))
    cnt = collections.defaultdict(collections.Counter)
    for row, g in key.items():
        cnt[g][MAP.get(last["r4_%02d" % int(row)]["value"], "(外す: 分からない・空)")] += 1
    for g in ("film", "control"):
        print(f"  {g}: " + " / ".join(f"{k} {v}" for k, v in cnt[g].most_common()))
    k1 = cnt["film"]["あたり+下絵"]; n1 = sum(v for k, v in cnt["film"].items() if not k.startswith("(外す"))
    k0 = cnt["control"]["あたり+下絵"]; n0 = sum(v for k, v in cnt["control"].items() if not k.startswith("(外す"))
    lo = beta.ppf(0.025, k1, n1 - k1 + 1) if k1 else 0.0
    p = fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")[1]
    print(f"  膜の組の的中 {k1}/{n1} = {k1 / n1:.3f}(95% 下限 {lo:.3f})/ 対照 {k0}/{n0} / 差 {k1 / n1 - k0 / n0:.3f} / 片側 Fisher p {p:.4f}")
    return k1, n1, lo


def main() -> int:
    print("[R4 の答えを census の鍵で引き直す]")
    rescore()
    c = db.connect()
    names = {(p, i): (n or "") for p, i, n in c.execute(
        "select l.page_path, l.layer_id, l.name from layers l join pages p using (page_path) where p.format='clip'").fetchall()}
    popk = {(r["page_path"], int(r["layer_id"])) for r in csv.DictReader(open(POP, encoding="utf-8"))}
    out, cnt, drop = [], collections.Counter(), collections.Counter()
    for r in csv.DictReader(open(PIX, encoding="utf-8")):
        k = (r["page_path"], int(r["layer_id"]))
        n = names.get(k)
        if n is None:
            drop["DB に層が無い"] += 1
            continue
        nr = name_rule(n)
        gt0 = int(r["n_px_a_gt0"] or 0)
        ratio = int(r["n_px_a_ge26"] or 0) / gt0 if gt0 > 0 else None
        film = "" if ratio is None else int(ratio < 0.5)
        v = nr
        if nr == "なし" and film == 1:
            v = F_LABEL
        inpop = int(k in popk)
        cnt[(v, inpop if nr == "なし" else "-")] += 1
        out.append(dict(page_path=k[0], layer_id=k[1], name_rule=nr,
                        a26_ratio="" if ratio is None else f"{ratio:.4f}", film=film,
                        name_rule_film=v, in_r4_population=inpop))
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"\n[全層] 全頁の画素のまとめた表の層 {len(out):,}(落とした: " + (", ".join(f"{k} {v}" for k, v in drop.items()) or "0") + ")")
    print("  name_rule_film × R4 の母集団に入るか(なし の層だけ。他は -):")
    for (v, ip), m in sorted(cnt.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        print(f"  {m:8,d}  {v} / {ip}")
    nf = sum(m for (v, ip), m in cnt.items() if v == F_LABEL)
    nf_in = cnt[(F_LABEL, 1)]
    print(f"  F が言う層 {nf:,}(うち確かめた母集団の中 {nf_in:,} / 外 {nf - nf_in:,})")
    print(f"\n出力: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
