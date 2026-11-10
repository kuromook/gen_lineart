"""Candidate detector for the author's 'mysterious solid black' in v4 sketch layers (2026-11-10). Not a column of record
(discipline 13): printed against the author's eye answers first. Per page: largest connected region of near-uniform dark
(gray<60 after a 9x9 opening) and its area share of the page; also the share of such regions that are flat (std<6)."""
import json, sys, cv2, numpy as np
from multiprocessing import Pool
R = "data/clip_pairs_v4/clip_pairs/"

def one(it):
    s = cv2.imread(R + it["images"][0]["src"], 0)
    m = cv2.morphologyEx((s < 60).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return it["id"], 0.0, 0.0
    k = 1 + int(np.argmax(st[1:, 4]))
    return it["id"], st[k, 4] / s.size, float(m.sum()) / s.size

if __name__ == "__main__":
    Q = json.load(open("data/sketch_eye_check_20261110/questions.json", encoding="utf-8"))
    with Pool(7) as p:
        rows = p.map(one, Q["items"])
    with open("data/sketch_eye_check_20261110/solid_blob.csv", "w") as f:
        f.write("item_id,largest_blob_frac,all_blob_frac\n")
        for r in rows:
            f.write("%s,%.6f,%.6f\n" % r)
    print(len(rows), "pages")
