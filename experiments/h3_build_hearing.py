"""聞き取り H3 の問いを作る(登録: doc/work_log.md 2026-11-27「登録 S3」確かめの段・「登録 S4」)。
S4(T1 × C θ0.50)の区分で「在る」10・「無い」10。各 10 のうち 5 は S2(T0 × K/L)から区分が変わったコマ(足りなければ在るだけ)。
H1/H1b の頁・H2 のコマは除く。代表の作品だけ。種 20261127。出力 data/h3_panel_sketch_20261127/(追跡しない)。
"""
import csv, glob, json, random, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from s1_cleanup_delta import find, shift  # noqa: E402
from s2_panel_delta import PAGES, SHAPES, s1_rows  # noqa: E402
from s3_binarize import binarize_sketch  # noqa: E402
from s1_cleanup_delta import binarize  # noqa: E402
from h2_build_hearing import SKIP_WORKS, _shift_gray, pick  # noqa: E402

OUT = ROOT / "data/h3_panel_sketch_20261127"
SEED = 20261127


def load(d):
    K = {}
    for f in glob.glob(str(ROOT / d / "*/s2_panels.csv")):
        for k in csv.DictReader(open(f, encoding="utf-8")):
            K[(k["line"], k["koma_id"])] = k
    return K


def main():
    S1 = s1_rows()
    q_by_line = {q["line"]: q for q, _ in S1.values()}
    s1_by_line = {q["line"]: s for q, s in S1.values()}
    h1 = list(csv.DictReader(open(ROOT / "data/h1_sketch_stage_20261120/selection.csv", encoding="utf-8")))
    skip_lines = {next(q["line"] for q, s in S1.values() if q["work_id"] == r["work_id"] and q["page"] == r["page"] and s["B_keep"] == r["B_keep"]) for r in h1}
    skip_lines |= {r["line"] for r in csv.DictReader(open(ROOT / "data/h1b_reask_20261126/selection.csv", encoding="utf-8"))}
    skip_koma = {(r["line"], r["koma_id"]) for r in csv.DictReader(open(ROOT / "data/h2_panel_sketch_20261127/selection.csv", encoding="utf-8"))}
    old, new = load("results/s2_full"), load("results/s4_full")
    cand = [k for key, k in new.items() if key not in skip_koma and k["line"] not in skip_lines and k["work_id"] not in SKIP_WORKS
            and q_by_line[k["line"]]["is_primary"] == "True" and k["kind"] != "線の少ないコマ"]
    for k in cand:
        k["_changed"] = old.get((k["line"], k["koma_id"]), {}).get("kind") != k["kind"]
    rng = random.Random(SEED)
    sel = []
    for cls, kind in (("在る", "下絵の在るコマ"), ("無い", "下絵の無いコマ")):
        ch = [k for k in cand if k["kind"] == kind and k["_changed"]]
        un = [k for k in cand if k["kind"] == kind and not k["_changed"]]
        got = pick(ch, 5, rng)
        got += pick(un, 10 - len(got), rng)
        print(f"{cls}: 候補 変わった {len(ch)} / 変わらない {len(un)} → 変わった {sum(k['_changed'] for k in got)}・計 {len(got)}")
        sel += [(cls, k) for k in got]
    rng.shuffle(sel)
    pages = {r["page_path"]: r for r in csv.DictReader(open(PAGES, encoding="utf-8"))}
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    items = []
    with open(OUT / "selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "instrument", "changed_from_s2", "work_id", "line", "koma_id", "C", "K_over_L", "keep", "added", "bbox"])
        for i, (cls, k) in enumerate(sel, 1):
            iid = f"q{i:02d}"
            q = q_by_line[k["line"]]; s = s1_by_line[k["line"]]
            dx, dy = int(s["B_dx"]), int(s["B_dy"])
            lab = cv2.imread(str(SHAPES / Path(pages[q["source_path"]]["out_file"]).name), cv2.IMREAD_UNCHANGED)
            inside = lab == int(k["koma_id"])
            ys, xs = np.nonzero(inside)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            m = inside[y0:y1, x0:x1]
            ln = cv2.imread(str(find(q, "line")), cv2.IMREAD_GRAYSCALE)
            sk = _shift_gray(cv2.imread(str(find(q, "sketch")), cv2.IMREAD_GRAYSCALE), dx, dy)
            Lb = binarize(find(q, "line"))[0]
            Sb = shift(binarize_sketch(find(q, "sketch"), "T1"), dx, dy)
            ov = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
            ov[Sb[y0:y1, x0:x1]] = (170, 170, 170)
            ov[Lb[y0:y1, x0:x1]] = (0, 0, 0)
            for name, im in (("sketch", cv2.cvtColor(sk[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)),
                             ("line", cv2.cvtColor(ln[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)), ("overlay", ov)):
                im = im.copy()
                im[~m] = (im[~m] * 0.25 + 255 * 0.75).astype(np.uint8)
                cv2.imwrite(str(OUT / "pages" / f"{iid}_{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            w.writerow([iid, cls, k["_changed"], k["work_id"], k["line"], k["koma_id"], k["C"], k["K_over_L"], k["keep"], k["added"], f"{x0},{y0},{x1},{y1}"])
            items.append({"id": iid, "version": "s2_panel_shapes_20261126", "images": [
                {"src": f"pages/{iid}_sketch.jpg", "caption": "下絵(このコマの所)"},
                {"src": f"pages/{iid}_line.jpg", "caption": "線画(このコマの所)", "target": True},
                {"src": f"pages/{iid}_overlay.jpg", "caption": "重ね(下絵=灰・線画=黒)"}]})
            print(f"  {iid} 器具={cls} 変わった={k['_changed']} {k['work_id']} {Path(k['line']).stem} koma{k['koma_id']} C {k['C']}")
    spec = {"interview_id": "s_h3_panel_sketch_20261127", "page_title": "コマの下絵 H3",
            "question": "このコマ(薄くしていない所)に、この線画の下絵はありますか?", "unit": "コマ",
            "decides": "薄い下絵も拾うように直した器具が、作者の目に合うかを数えるのに使います",
            "choices": [{"value": "present", "label": "ある", "help": "このコマの絵の下絵が下絵の層にある"},
                        {"value": "partial", "label": "一部だけ", "help": "コマの一部(人物の一部など)にだけある"},
                        {"value": "redrawn", "label": "あるが、線画で描き直した", "help": "下絵はあるが、線画は形や構図を変えている"},
                        {"value": "absent", "label": "無い", "help": "このコマの下絵は下絵の層に無い"}],
            "items": items}
    (OUT / "questions.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"問い {len(items)} 件")


if __name__ == "__main__":
    main()
