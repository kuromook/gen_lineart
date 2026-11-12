"""Show the author ~20 samples of the delivered v5 pairs before any number is read (author 2026-11-12: "数字だけでまだ
判断しない 20点程度でいいのでUIにして見せてほしい"). Pages come from the v5 measured population (1,281), grouped by how
the delivery changed them (image_diff_v4_vs_v5.csv), seeded. Where a side changed, the v4 image of that side is shown
beside it for comparison. Output under data/ (manuscript names; not tracked)."""
import csv, json, os, random, sys
sys.path.insert(0, "tools/evaluation")
from pair_feature_tiles import load_jobs

OUT = "data/v5_sample_view_20261112"
os.makedirs(OUT, exist_ok=True)
rng = random.Random(20261112)
jobs = load_jobs("data/clip_pairs_v5", qc="clip_pairs_qc_v5.csv")
keyset = {k for k, _, _ in jobs}
paths = {k: (sp, lp) for k, sp, lp in jobs}
diff = {}
for r in csv.DictReader(open("data/clip_pairs_v5/clip_pairs/image_diff_v4_vs_v5.csv", encoding="utf-8")):
    k = f"{r['work_id']}|{r['slug']}|{r['page']}"
    if k in keyset:
        diff.setdefault(k, {})[r["role"]] = r


def ratio(r):
    try:
        return float(r["ink_mean_v5"]) / float(r["ink_mean_v4"])
    except (ValueError, ZeroDivisionError):
        return None


eye = {json.loads(l)["item_id"]: json.loads(l) for l in open("data/sketch_eye_check_20261110/answers.jsonl", encoding="utf-8")}
v4keys = [k for k, _, _ in load_jobs("data/clip_pairs_v4")]
pid = {k: f"p{i:04d}" for i, k in enumerate(v4keys)}
beta = [k for k in keyset if pid[k] in eye and eye[pid[k]].get("item_version") == "clip_pairs_v4"
        and "ベタ" in eye[pid[k]].get("comment", "") and diff[k]["sketch"]["same_bytes"] != "1"]

groups = [
    ("下絵の白の化けが直った頁(11-10 に謎のベタと答えた頁)", beta, 4),
    ("下絵が大きく薄くなった頁(色の線の下絵)", [k for k in keyset if (ratio(diff[k]["sketch"]) or 1) < 0.5], 4),
    ("線画が大きく薄くなった頁(灰色・色の塗りの層が線画に入っている)", [k for k in keyset if (ratio(diff[k]["line"]) or 1) < 0.5], 4),
    ("線画が少し変わった頁(墨が v4 の 0.8〜0.95)", [k for k in keyset if 0.8 <= (ratio(diff[k]["line"]) or 1) < 0.95], 3),
    ("v4 から変わっていない頁", [k for k in keyset if diff[k]["line"]["same_bytes"] == "1" and diff[k]["sketch"]["same_bytes"] == "1"], 5),
]
spec = {
    "interview_id": "pair_signal_v5_sample_view_20261112",
    "page_title": "v5 の対の見本 2026-11-12",
    "question": "この v5 の対は、下絵と線画の対として使えそうですか?",
    "unit": "頁の対(v5)",
    "decides": "表の数を読む前に、納品物の中身を見て、解釈に進んでよいかを決める",
    "choices": [
        {"value": "ok", "label": "使えそう", "help": "下絵も線画も、対として見ておかしくない"},
        {"value": "sketch_bad", "label": "下絵がおかしい", "help": "何がおかしいかを一言欄に"},
        {"value": "line_bad", "label": "線画がおかしい", "help": "何がおかしいかを一言欄に"},
        {"value": "worse", "label": "v4 より悪くなった", "help": "v4 の絵が並んでいる頁で、v4 の方がよい"},
    ],
    "items": [],
}
seen = set()
for gname, pool, n in groups:
    pool = sorted(set(pool) - seen)
    rng.shuffle(pool)
    print(f"{gname}: pool {len(pool)}, take {min(n, len(pool))}")
    for k in pool[:n]:
        seen.add(k)
        sp, lp = paths[k]
        rel = lambda p: os.path.relpath(p, "data")
        imgs = [{"src": rel(sp), "caption": "v5 下絵", "target": True},
                {"src": rel(lp), "caption": "v5 線画", "target": True}]
        if diff[k]["sketch"]["same_bytes"] != "1":
            imgs.append({"src": rel(sp.replace("clip_pairs_v5", "clip_pairs_v4")), "caption": "v4 下絵(比べ用)"})
        if diff[k]["line"]["same_bytes"] != "1":
            imgs.append({"src": rel(lp.replace("clip_pairs_v5", "clip_pairs_v4")), "caption": "v4 線画(比べ用)"})
        w, slug, page = k.split("|")
        spec["items"].append({"id": f"s{len(spec['items']):02d}", "version": "clip_pairs_v5", "key": k,
                              "note": f"{gname} / {w} / {page}", "images": imgs})
for it in spec["items"]:
    for im in it["images"]:
        assert os.path.exists("data/" + im["src"]), im["src"]
json.dump(spec, open(f"{OUT}/questions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(spec["items"]), "items")
