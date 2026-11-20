# -*- coding: utf-8 -*-
"""R4 の採点(登録 config/prereg_r4_film_rule_nashi_20261120.md の通り)。答えが30行そろった後に鍵を開く。
使い方: ./venv/bin/python tools/r4_score.py   出力は印字だけ(原稿の名前を含むので data/ に控える)"""
import csv, io, json
from collections import Counter, defaultdict
from scipy.stats import fisher_exact, beta

D = "data/census"
ANS = D + "/r4_film_nashi_images_20261120/r4_answers.jsonl"
MAP = {"atari": "あたり+下絵", "shitae": "あたり+下絵", "line": "清書", "__none__": "段ではない"}

def cp_lower(k, n, a=0.05):
    return 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)

def main():
    last = {}
    for l in io.open(ANS, encoding="utf-8"):
        r = json.loads(l); last[r["item_id"]] = r          # 最新の答え(選び直しは 0 件)
    tab = {r["row"]: r for r in csv.DictReader(io.open(D + "/r4_film_nashi_hearing_table_20261120.csv", encoding="utf-8"))}
    key = {r["row"]: r for r in csv.DictReader(io.open(D + "/r4_film_nashi_hearing_key_20261120.csv", encoding="utf-8"))}
    assert len(last) == len(tab) == len(key) == 30
    rows, drop = [], Counter()
    for p, t in tab.items():
        k = key[p]; assert (k["page_path"], k["layer_id"]) == (t["page_path"], t["layer_id"])
        a = last["r4_%02d" % int(p)]
        st = MAP.get(a["value"])
        g = "膜" if k["group"] == "film" else "対照"
        hid = not (t["self_shown"] == "1" and t["ancestors_shown"] == "1")
        rec = dict(pos=int(p), group=g, ans=a["label"], stage=st, comment=a["comment"], work=t["work_dir"].split("\\")[-1],
                   year=t["year_label"], hidden=hid, name=t["layer_name"])
        if st is None:
            drop[(g, a["label"])] += 1; rec["stage"] = "(外す)"
        rows.append(rec)
    print("外した行(分からない・空):", dict(drop) or "なし")
    for g in ("膜", "対照"):
        print(g, "の段の内訳:", dict(Counter(r["stage"] for r in rows if r["group"] == g)),
              "/ 答え:", dict(Counter(r["ans"] for r in rows if r["group"] == g)))
    def kn(g, f=lambda r: True):
        rr = [r for r in rows if r["group"] == g and r["stage"] != "(外す)" and f(r)]
        return sum(r["stage"] == "あたり+下絵" for r in rr), len(rr)
    kf, nf = kn("膜"); kc, nc = kn("対照")
    rf, rc = kf / nf, kc / nc
    if nf < 15: c1 = "判定保留(15行未満)"
    elif nf >= 20: c1 = "通る" if kf >= 16 else "落ちる"
    else: c1 = "通る" if rf >= 0.80 else "落ちる"
    print("C1: 膜 あたり+下絵 %d/%d = %.3f(棒 16/20 か 0.80)、95%% 下限 %.3f → %s" % (kf, nf, rf, cp_lower(kf, nf), c1))
    p = fisher_exact([[kf, nf - kf], [kc, nc - kc]], alternative="greater")[1]
    c2 = "通る" if rf - rc >= 0.30 and p < 0.05 else "落ちる"
    print("C2: 対照 %d/%d = %.3f、差 %.3f(棒 0.30)、片側 Fisher p = %.4f(棒 0.05)→ %s" % (kc, nc, rc, rf - rc, p, c2))
    for lab, f in (("表示", lambda r: not r["hidden"]), ("非表示", lambda r: r["hidden"])):
        print(lab, "膜 %d/%d 対照 %d/%d" % (kn("膜", f) + kn("対照", f)))
    by = defaultdict(list)
    for r in rows: by[(r["year"])].append(r)
    for y in sorted(by):
        print("年代", y, " ".join("%s:%s" % (r["group"], r["stage"]) for r in by[y]))
    w = Counter(r["work"] for r in rows if r["group"] == "膜"); print("膜の組の作品ごとの層数:", sorted(w.values(), reverse=True))
    print("--- 全行")
    for r in sorted(rows, key=lambda r: (r["group"], r["pos"])):
        print("%2d %s %-6s %-10s %s %s / %s / 名 %s / %s" % (r["pos"], r["group"], r["stage"], r["ans"], "非表示" if r["hidden"] else "表示",
              r["year"], r["work"], r["name"], r["comment"]))

if __name__ == "__main__":
    main()
