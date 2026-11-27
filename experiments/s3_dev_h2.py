"""登録 S3 の開発の段: H2 の20コマで、下絵の二値化の腕ごとに K/L の区分を出し、作者の答えと照らす。"""
import csv, json, sys
from pathlib import Path
import cv2, numpy as np
from skimage.morphology import skeletonize
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from s1_cleanup_delta import find, shift  # noqa: E402
from s2_panel_delta import NO_SKETCH, PAGES, SHAPES, s1_rows  # noqa: E402
from s3_binarize import binarize_sketch  # noqa: E402

H2 = ROOT / "data/h2_panel_sketch_20261127"


def main():
    S1 = s1_rows()
    q_by_line = {q["line"]: q for q, _ in S1.values()}
    s1_by_line = {q["line"]: s for q, s in S1.values()}
    pages = {r["page_path"]: r for r in csv.DictReader(open(PAGES, encoding="utf-8"))}
    L_of = {}
    import glob
    for f in glob.glob(str(ROOT / "results/s2_full/*/s2_panels.csv")):
        for k in csv.DictReader(open(f, encoding="utf-8")):
            L_of[(k["line"], k["koma_id"])] = int(k["L"])
    sel = list(csv.DictReader(open(H2 / "selection.csv", encoding="utf-8")))
    ans = {json.loads(l)["item_id"]: json.loads(l)["value"] for l in open(H2 / "s_h2_panel_sketch_20261127_answers.jsonl", encoding="utf-8")}
    arms = ["T0", "T1", "T2"]
    hits = {a: 0 for a in arms}
    rows = []
    for r in sel:
        q, s = q_by_line[r["line"]], s1_by_line[r["line"]]
        lab = cv2.imread(str(SHAPES / Path(pages[q["source_path"]]["out_file"]).name), cv2.IMREAD_UNCHANGED)
        inside = lab == int(r["koma_id"])
        L = L_of[(r["line"], r["koma_id"])]
        out = {}
        for a in arms:
            Sb = shift(binarize_sketch(find(q, "sketch"), a), int(s["B_dx"]), int(s["B_dy"]))
            K = int((skeletonize(Sb) & inside).sum())
            cls = "無い" if K / L < NO_SKETCH else "在る"
            out[a] = (round(K / L, 3), cls)
            hits[a] += (cls == "在る" and ans[r["item_id"]] == "present") or (cls == "無い" and ans[r["item_id"]] == "absent")
        rows.append((r["item_id"], r["work_id"], ans[r["item_id"]], out))
        print(r["item_id"], f"{r['work_id']:22s} 作者 {ans[r['item_id']]:8s}", " ".join(f"{a} {out[a][0]:7.3f} {out[a][1]}" for a in arms))
    print("厳しい版の的中:", hits)
    with open(ROOT / "results/s3_dev_h2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["item_id", "work_id", "author"] + [f"{a}_{x}" for a in arms for x in ("KL", "cls")])
        for i, wk, au, o in rows:
            w.writerow([i, wk, au] + [v for a in arms for v in o[a]])


if __name__ == "__main__":
    main()
