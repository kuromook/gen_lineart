"""聞き取り H1 の問いを作る(登録: doc/work_log.md 2026-11-20「聞き取り H1 の登録」)。

出力は data/h1_sketch_stage_20261120/(追跡しない。原稿の画素と名前を含む):
  questions.json / pages/<id>_{sketch,line,overlay}.jpg / selection.csv(保留か対照かの答え合わせ用。画面には出さない)
"""
import csv
import glob
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from s1_cleanup_delta import binarize, find, population, shift  # noqa: E402

OUT = ROOT / "data/h1_sketch_stage_20261120"
SEED = 20261120
STRATA = [("2015〜2016", ("2015", "2016"), 3), ("2017", ("2017",), 3), ("2018", ("2018",), 4),
          ("2019〜2019b", ("2019", "2019b"), 3), ("2022〜2024", ("2022", "2022b", "2023", "2023b", "2024", "2024b", "2024c"), 3)]
CONTROL_YEARS = ["2016", "2017", "2022b", "2024b"]


def year(r):
    return r["work_id"].split("_")[1]


def pick_round_robin(rows, n, rng):
    by = {}
    for r in rows:
        by.setdefault(r["work_id"], []).append(r)
    works = sorted(by)
    rng.shuffle(works)
    for w in works:
        rng.shuffle(by[w])
    out = []
    while len(out) < n and any(by.values()):
        for w in works:
            if by[w] and len(out) < n:
                out.append(by[w].pop())
    return out


def main():
    # 11-21 修正: (work_id, page) は一意でない(QC 表で 91 鍵・214 行が重なる)。旧版は dict の最後の行を引き、
    # h04・h18 で S1 の行と別の頁(母集団の外)の絵を出した。S1 の行は population() を (slug, page) で並べた順なので、
    # 同じ並びで行ごとに対応させ、頁名の一致を確かめる
    pop = population()
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results/s1_full/*/s1_pages.csv"))):
        S = list(csv.DictReader(open(f, encoding="utf-8")))
        Pw = sorted([r for r in pop if r["work_id"] == Path(f).parent.name], key=lambda r: (r["slug"], r["page"]))
        assert [r["page"] for r in Pw] == [r["page"] for r in S], f
        for s, q in zip(S, Pw):
            s["_qc"] = q
        rows += S
    prim = [r for r in rows if r["_qc"]["is_primary"] == "True"]
    held = [r for r in prim if r["hold"] == "True" and r["N2_keep"] not in ("nan", "")]
    used = [r for r in prim if r["hold"] != "True"]
    rng = random.Random(SEED)
    sel = []
    for name, ys, n in STRATA:
        got = pick_round_robin([r for r in held if year(r) in ys], n, rng)
        print(f"保留 {name}: 候補 {sum(year(r) in ys for r in held)} → {len(got)}")
        sel += [("held", name, r) for r in got]
    for y in CONTROL_YEARS:
        got = pick_round_robin([r for r in used if year(r) == y], 1, rng)
        print(f"対照 {y}: 候補 {sum(year(r) == y for r in used)} → {len(got)}")
        sel += [("control", y, r) for r in got]
    rng.shuffle(sel)

    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    items = []
    with open(OUT / "selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "kind", "stratum", "work_id", "page", "B_keep", "N2_keep", "B_dx", "B_dy"])
        for i, (kind, stratum, r) in enumerate(sel, 1):
            iid = f"h{i:02d}"
            q = r["_qc"]
            sk_src, ln_src = find(q, "sketch"), find(q, "line")
            shutil.copyfile(sk_src, OUT / "pages" / f"{iid}_sketch.jpg")
            shutil.copyfile(ln_src, OUT / "pages" / f"{iid}_line.jpg")
            Lb, _, _ = binarize(ln_src)
            Sb, _, _ = binarize(sk_src)
            Sb = shift(Sb, int(r["B_dx"]), int(r["B_dy"]))
            ov = np.full(Lb.shape + (3,), 255, np.uint8)
            ov[Sb] = (170, 170, 170)
            ov[Lb] = (0, 0, 0)
            cv2.imwrite(str(OUT / "pages" / f"{iid}_overlay.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 92])
            w.writerow([iid, kind, stratum, r["work_id"], r["page"], r["B_keep"], r["N2_keep"], r["B_dx"], r["B_dy"]])
            items.append({"id": iid, "version": "clip_pairs_v6", "images": [
                {"src": f"pages/{iid}_sketch.jpg", "caption": "下絵"},
                {"src": f"pages/{iid}_line.jpg", "caption": "線画", "target": True},
                {"src": f"pages/{iid}_overlay.jpg", "caption": "重ね(下絵=灰・線画=黒)"}]})
            print(f"  {iid} {kind:7s} {stratum:12s} {r['work_id']} {r['page']}")
    spec = {
        "interview_id": "s_h1_sketch_stage_20261120",
        "page_title": "下絵と線画 H1",
        "question": "この下絵は、この線画を描くときに下に敷いていた下絵ですか?",
        "unit": "頁",
        "decides": "下絵と線画が合わない頁を、理由ごとに数えるのに使います",
        "choices": [
            {"value": "traced", "label": "敷いていた・おおむね下絵どおり", "help": "線画は下絵の形をなぞっている"},
            {"value": "rough", "label": "敷いていた・下絵が粗い", "help": "下絵は大まかで、形は線画で決めた"},
            {"value": "redrawn", "label": "敷いていた・線画で描き直した", "help": "下絵はあったが、線画で変えた"},
            {"value": "not_its_sketch", "label": "この線画の下絵ではない", "help": "別の段の下絵・別の絵など"}],
        "items": items}
    (OUT / "questions.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"問い {len(items)} 件: {(OUT / 'questions.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
