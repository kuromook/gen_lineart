"""登録 S4 の開発の段: H2 の20コマで、二値化の腕 × θ ごとに C(下絵から R2 以内の線画の骨格の割合)の区分を出す。"""
import csv, glob, json, sys
from pathlib import Path
import cv2, numpy as np
from skimage.morphology import skeletonize
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from s1_cleanup_delta import FILL_PAD, R2, binarize, dist_to_ink, fill_mask, find, shift  # noqa: E402
from s2_panel_delta import FEW_LINE, PAGES, SHAPES, s1_rows  # noqa: E402
from s3_binarize import binarize_sketch  # noqa: E402

H2 = ROOT / "data/h2_panel_sketch_20261127"
ARMS, THETAS = ["T0", "T1", "T2"], [0.30, 0.50]


def main():
    S1 = s1_rows()
    q_by_line = {q["line"]: q for q, _ in S1.values()}
    s1_by_line = {q["line"]: s for q, s in S1.values()}
    pages = {r["page_path"]: r for r in csv.DictReader(open(PAGES, encoding="utf-8"))}
    sel = list(csv.DictReader(open(H2 / "selection.csv", encoding="utf-8")))
    ans = {json.loads(l)["item_id"]: json.loads(l)["value"] for l in open(H2 / "s_h2_panel_sketch_20261127_answers.jsonl", encoding="utf-8")}
    hits = {(a, t): 0 for a in ARMS for t in THETAS}
    cls_of = {}
    rows = []
    for r in sel:
        q, s = q_by_line[r["line"]], s1_by_line[r["line"]]
        lab = cv2.imread(str(SHAPES / Path(pages[q["source_path"]]["out_file"]).name), cv2.IMREAD_UNCHANGED)
        Lb, _, _ = binarize(find(q, "line"))
        F = fill_mask(Lb)
        Fpad = cv2.dilate(F.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FILL_PAD + 1,) * 2)) > 0
        Lsk = skeletonize(Lb) & ~Fpad & (lab == int(r["koma_id"]))
        ly, lx = np.nonzero(Lsk)
        Cs = {}
        for a in ARMS:
            Sb = shift(binarize_sketch(find(q, "sketch"), a), int(s["B_dx"]), int(s["B_dy"]))
            Cs[a] = float((dist_to_ink(Sb)[ly, lx] <= R2).mean()) if len(ly) else float("nan")
        au = ans[r["item_id"]]
        for a in ARMS:
            for t in THETAS:
                c = "少" if len(ly) < FEW_LINE else ("無い" if Cs[a] < t else "在る")
                cls_of[(r["item_id"], a, t)] = c
                hits[(a, t)] += (c == "在る" and au == "present") or (c == "無い" and au == "absent")
        rows.append([r["item_id"], r["work_id"], au, len(ly)] + [round(Cs[a], 3) for a in ARMS])
        print(r["item_id"], f"{r['work_id']:22s} 作者 {au:8s} L {len(ly):6d}", " ".join(f"{a} C {Cs[a]:.3f}" for a in ARMS))
    print("厳しい版の的中:", {f"{a}/{t}": v for (a, t), v in hits.items()})
    t0_absent = [r["item_id"] for r in sel if ans[r["item_id"]] == "absent" and r["instrument"] == "無い"]
    for a in ARMS:
        for t in THETAS:
            c1 = all(cls_of[(i, a, t)] == "在る" for i in ("p02", "p07"))
            c2 = all(cls_of[(i, a, t)] == "無い" for i in t0_absent)
            bad = [i for i in t0_absent if cls_of[(i, a, t)] != "無い"] + [i for i in ("p02", "p07") if cls_of[(i, a, t)] != "在る"]
            print(f"  {a} θ{t}: (1) {c1} (2) {c2} 的中 {hits[(a, t)]} 外れの要所 {bad}")
    with open(ROOT / "results/s4_dev_h2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["item_id", "work_id", "author", "L"] + [f"C_{a}" for a in ARMS]); w.writerows(rows)


if __name__ == "__main__":
    main()
