"""H1b: H1 の h04・h18 を正しい頁で聞き直す(登録: doc/work_log.md 2026-11-26)。h1_build_hearing の修正後の対応で頁を引く。"""
import csv, glob, json, shutil, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from s1_cleanup_delta import binarize, find, population, shift  # noqa: E402
OUT = ROOT / "data/h1b_reask_20261126"
H1 = ROOT / "data/h1_sketch_stage_20261120"
WANT = {"h04": "021_2018_comics4_mars_v2_mars_manuscript_page0006_line.jpg",
        "h18": "069_2024b_fitness_kurip_page0030_line.jpg"}


def main():
    pop = population()
    rows = {}
    for f in sorted(glob.glob(str(ROOT / "results/s1_full/*/s1_pages.csv"))):
        S = list(csv.DictReader(open(f, encoding="utf-8")))
        Pw = sorted([r for r in pop if r["work_id"] == Path(f).parent.name], key=lambda r: (r["slug"], r["page"]))
        assert [r["page"] for r in Pw] == [r["page"] for r in S], f
        for s, q in zip(S, Pw):
            rows[q["line"]] = (s, q)
    sel = {r["item_id"]: r for r in csv.DictReader(open(H1 / "selection.csv", encoding="utf-8"))}
    spec = json.loads((H1 / "questions.json").read_text(encoding="utf-8"))
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    items = []
    with open(OUT / "selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "kind", "stratum", "work_id", "page", "line", "source_path", "B_keep", "N2_keep", "B_dx", "B_dy"])
        for old, line in WANT.items():
            s, q = rows[line]
            o = sel[old]
            assert (s["B_keep"], s["B_dx"], s["B_dy"]) == (o["B_keep"], o["B_dx"], o["B_dy"]), old  # H1 で選んだ行と同じ
            iid = old + "b"
            sk, ln = find(q, "sketch"), find(q, "line")
            shutil.copyfile(sk, OUT / "pages" / f"{iid}_sketch.jpg")
            shutil.copyfile(ln, OUT / "pages" / f"{iid}_line.jpg")
            Lb, _, _ = binarize(ln)
            Sb = shift(binarize(sk)[0], int(s["B_dx"]), int(s["B_dy"]))
            ov = np.full(Lb.shape + (3,), 255, np.uint8)
            ov[Sb] = (170, 170, 170)
            ov[Lb] = (0, 0, 0)
            cv2.imwrite(str(OUT / "pages" / f"{iid}_overlay.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 92])
            w.writerow([iid, o["kind"], o["stratum"], q["work_id"], q["page"], line, q["source_path"], s["B_keep"], s["N2_keep"], s["B_dx"], s["B_dy"]])
            items.append({"id": iid, "version": "clip_pairs_v6", "images": [
                {"src": f"pages/{iid}_sketch.jpg", "caption": "下絵"},
                {"src": f"pages/{iid}_line.jpg", "caption": "線画", "target": True},
                {"src": f"pages/{iid}_overlay.jpg", "caption": "重ね(下絵=灰・線画=黒)"}]})
            print(iid, o["kind"], q["source_path"])
    spec.update(interview_id="s_h1b_reask_20261126", page_title="下絵と線画 H1b(聞き直し2頁)", items=items)
    (OUT / "questions.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
