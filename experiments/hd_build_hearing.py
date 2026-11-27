"""聞き取り HD の問いを作る(登録: doc/work_log.md 2026-11-27「登録 HD」)。線画の骨格の破片1本ごとに、
下絵の墨までの距離 d の中央値で層に分けて引き、作者に「赤い線は下絵のどの線から来たか」を問う。
出力 data/hd_boundary_20261127/(追跡しない)。"""
import csv, glob, json, random, sys
from pathlib import Path
import cv2, numpy as np
from skimage.morphology import skeletonize
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from s1_cleanup_delta import FILL_PAD, binarize, dist_to_ink, fill_mask, find, shift  # noqa: E402
from s2_panel_delta import PAGES, SHAPES, s1_rows  # noqa: E402
from s3_binarize import binarize_sketch  # noqa: E402
from h2_build_hearing import SKIP_WORKS, _shift_gray  # noqa: E402

OUT = ROOT / "data/hd_boundary_20261127"
SEED = 20261127
BINS = [(0, 2, 3), (2, 4, 3), (4, 6, 3), (6, 10, 3), (10, 20, 3), (20, 40, 3), (40, 1e9, 2)]
MIN_LEN, SAME_BIN, HALF = 80, 0.8, 256  # 11-27 改: 40 では髪の束・影の線の切れ端になり問いにくい
NEED = 10  # 層ごとに候補をこれだけ(作品の違うもの)集めたら止める(11-27 改: 6 では 32コマで止まり偏った)


def bin_of(d):
    for i, (lo, hi, _) in enumerate(BINS):
        if lo <= d < hi or (i == 0 and d == 0):
            return i
    return len(BINS) - 1


def segments(sk):
    """骨格を交点(8近傍が3以上)で切った連結の破片。"""
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    nb = cv2.filter2D(sk.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    body = sk & (nb <= 2)
    n, lab = cv2.connectedComponents(body.astype(np.uint8), connectivity=8)
    return n, lab


def main():
    S1 = s1_rows()
    q_by_line = {q["line"]: q for q, _ in S1.values()}
    s1_by_line = {q["line"]: s for q, s in S1.values()}
    asked = set()
    for d in ("hb_why_changed_20261128", "hb2_why_changed_20261127"):
        asked |= {(r["line"], r["koma_id"]) for r in csv.DictReader(open(ROOT / "data" / d / "selection.csv", encoding="utf-8"))}
    K = []
    for f in sorted(glob.glob(str(ROOT / "results/s4_full/*/s2_panels.csv"))):
        K += list(csv.DictReader(open(f, encoding="utf-8")))
    drop = {"下絵が無い・線が少ない": 0, "保留": 0, "二重の組": 0, "代表でない頁": 0, "HB・HB2 のコマ": 0}
    keep = []
    for k in K:
        if k["kind"] != "下絵の在るコマ": drop["下絵が無い・線が少ない"] += 1
        elif k["hold"] == "True": drop["保留"] += 1
        elif k["work_id"] in SKIP_WORKS: drop["二重の組"] += 1
        elif q_by_line[k["line"]]["is_primary"] != "True": drop["代表でない頁"] += 1
        elif (k["line"], k["koma_id"]) in asked: drop["HB・HB2 のコマ"] += 1
        else: keep.append(k)
    print(f"コマ {len(K)} → 使う {len(keep)} / 落とした {drop}")
    rng = random.Random(SEED)
    rng.shuffle(keep)
    keep.sort(key=lambda k: k["line"])  # 頁ごとにまとめて計算を使い回す
    rng2 = random.Random(SEED + 1); pg = sorted({k['line'] for k in keep}); rng2.shuffle(pg); order = {l: i for i, l in enumerate(pg)}
    keep.sort(key=lambda k: order[k["line"]])
    pages = {r["page_path"]: r for r in csv.DictReader(open(PAGES, encoding="utf-8"))}
    cand = [[] for _ in BINS]
    seen_panels = 0
    cache = {}
    for k in keep:
        if all(len({c["work_id"] for c in cb}) >= NEED for cb in cand):
            break
        seen_panels += 1
        q = q_by_line[k["line"]]; s = s1_by_line[k["line"]]
        dx, dy = int(s["B_dx"]), int(s["B_dy"])
        lab = cv2.imread(str(SHAPES / Path(pages[q["source_path"]]["out_file"]).name), cv2.IMREAD_UNCHANGED)
        inside = lab == int(k["koma_id"])
        if k["line"] not in cache:  # 同じ頁のコマは頁の計算を使い回す
            Lb = binarize(find(q, "line"))[0]
            Sb = shift(binarize_sketch(find(q, "sketch"), "T1"), dx, dy)
            F = fill_mask(Lb)
            Fpad = cv2.dilate(F.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FILL_PAD + 1,) * 2)) > 0
            cache.clear(); cache[k["line"]] = (Lb, Fpad, dist_to_ink(Sb))
        Lb, Fpad, dSfull = cache[k["line"]]
        ys_, xs_ = np.nonzero(inside)
        win = (slice(ys_.min(), ys_.max() + 1), slice(xs_.min(), xs_.max() + 1))
        sk = skeletonize(Lb[win]) & ~Fpad[win] & inside[win]
        dS = dSfull[win]  # 頁全体の下絵の墨まで(コマの外の墨も含める。S1 と同じ)
        n, sl = segments(sk)
        got = set()
        for i in rng.sample(range(1, n), n - 1):
            yy, xx = np.nonzero(sl == i)
            if len(yy) < MIN_LEN:
                continue
            d = dS[yy, xx]
            b = bin_of(float(np.median(d)))
            if b in got or np.mean([bin_of(float(v)) == b for v in d]) < SAME_BIN:
                continue
            got.add(b)
            cand[b].append(dict(work_id=k["work_id"], line=k["line"], koma_id=k["koma_id"], d_med=round(float(np.median(d)), 2),
                                length=len(yy), ys=(yy + win[0].start).tolist(), xs=(xx + win[1].start).tolist(), dx=dx, dy=dy))
    print(f"見たコマ {seen_panels}")
    sel = []
    used_panels = set()  # 11-27 改: 1コマから1本まで(初版は 20本が約11コマに偏った)
    for (lo, hi, n_), cb in sorted(zip(BINS, cand), key=lambda t: len(t[1])):
        by = {}
        for c in cb:
            if (c["line"], c["koma_id"]) not in used_panels:
                by.setdefault(c["work_id"], []).append(c)
        works = sorted(by); rng.shuffle(works)
        got = [by[w][0] for w in works[:n_]]
        used_panels |= {(c["line"], c["koma_id"]) for c in got}
        print(f"層 {lo}〜{hi if hi < 1e8 else ''}px: 候補 {len(cb)}(作品 {len(by)})→ {len(got)}")
        sel += [((lo, hi), c) for c in got]
    sel.sort(key=lambda t: t[0][0]); rng.shuffle(sel)
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    items = []
    with open(OUT / "selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "bin_lo", "bin_hi", "work_id", "line", "koma_id", "d_med", "length", "dx", "dy"])
        for i, ((lo, hi), c) in enumerate(sel, 1):
            iid = f"d{i:02d}"
            q = q_by_line[c["line"]]
            ln = cv2.imread(str(find(q, "line")), cv2.IMREAD_GRAYSCALE)
            sk = _shift_gray(cv2.imread(str(find(q, "sketch")), cv2.IMREAD_GRAYSCALE), c["dx"], c["dy"])
            ys, xs = np.array(c["ys"]), np.array(c["xs"])
            cy, cx = int(np.median(ys)), int(np.median(xs))
            H, W = ln.shape
            y0 = min(max(cy - HALF, 0), max(H - 2 * HALF, 0)); x0 = min(max(cx - HALF, 0), max(W - 2 * HALF, 0))
            win = (slice(y0, y0 + 2 * HALF), slice(x0, x0 + 2 * HALF))
            ov = cv2.cvtColor(ln[win], cv2.COLOR_GRAY2BGR)
            skw = sk[win]
            ov = np.minimum(ov, cv2.cvtColor(((skw.astype(np.float32) * 0.35 + 255 * 0.65)).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            mark = np.zeros(ln.shape, np.uint8); mark[ys, xs] = 1
            mark = cv2.dilate(mark, np.ones((5, 5), np.uint8))[win] > 0
            ov[mark] = (0, 0, 230)
            r = int(max(np.ptp(ys), np.ptp(xs)) / 2 + 24)
            cv2.circle(ov, (cx - x0, cy - y0), r, (0, 0, 230), 1, cv2.LINE_AA)  # 場所を見つけやすく(線に重ねない)
            for nm, im in (("sketch", skw), ("line", ln[win]), ("marked", ov)):
                cv2.imwrite(str(OUT / "pages" / f"{iid}_{nm}.png"), im)
            w.writerow([iid, lo, hi if hi < 1e8 else "", c["work_id"], c["line"], c["koma_id"], c["d_med"], c["length"], c["dx"], c["dy"]])
            items.append({"id": iid, "version": "s4_t1_hd", "images": [
                {"src": f"pages/{iid}_sketch.png", "caption": "下絵(線画に位置を合わせた)"},
                {"src": f"pages/{iid}_line.png", "caption": "線画"},
                {"src": f"pages/{iid}_marked.png", "caption": "重ね: 下絵=灰 線画=黒 問う線=赤", "target": True}]})
            print(f"  {iid} {lo}〜{hi if hi < 1e8 else ''} {c['work_id']} {Path(c['line']).stem} koma{c['koma_id']} d中央 {c['d_med']} 長さ {c['length']}")
    spec = {"interview_id": "s_hd_boundary_20261127", "page_title": "線の出どころ HD", "initial_view": "native",
            "question": "赤い線は、下絵のどの線から来た線ですか?", "unit": "線(赤の1本)",
            "decides": "線画の線を「下絵をなぞった・下絵から変えた・新しく描いた」に分ける境目を、作者の判断で決めるのに使います。わざとか、ずれかは問いません",
            "choices": [{"value": "traced", "label": "下絵の線をなぞった", "help": "同じ線・ほぼ同じ場所(わずかなずれはあっても)"},
                        {"value": "moved", "label": "下絵の線を元に、場所や形を変えて引いた", "help": "元になった下絵の線はあるが、ずらした・形を変えた"},
                        {"value": "new", "label": "下絵に元の線が無い", "help": "新しく描いた線"}],
            "items": items}
    (OUT / "questions.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"問い {len(items)} 件")


if __name__ == "__main__":
    main()
