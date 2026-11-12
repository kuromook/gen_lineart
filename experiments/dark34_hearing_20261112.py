"""Foundation request 2026-11-12 (reply_d_v5_dark_34_hearing): show the author the 34 v5 pages where the dark-region
rule still fires (pairs_v5_solid_dark_candidates_34_20261112.csv), in 2 batches of 17, v5 images, the SAME question and
six choices as the 11-10 / v5 eye check (user 11-12 「たさない」: the two added choices stay with Track Q).
All 34 are rule-positive, so answers measure hits only, not misses. Ids d00..d33 (seeded order), version clip_pairs_v5."""
import csv, glob, json, os, random, sys
D = "data/sketch_eye_check_20261110"
ROOT = "data/clip_pairs_v5/clip_pairs"
b = int(sys.argv[1])
rows = list(csv.DictReader(open("inbox/pairs_v5_solid_dark_candidates_34_20261112.csv", encoding="utf-8")))
random.Random(20261112).shuffle(rows)
qc = {f"{r['work_id']}|{r['slug']}|{r['page']}": r for r in csv.DictReader(open(f"{ROOT}/clip_pairs_qc_v5.csv", encoding="utf-8"))}
idx = {os.path.basename(p): p for p in glob.glob(f"{ROOT}/**/*.jpg", recursive=True)}
spec = json.load(open(f"{D}/questions.json", encoding="utf-8"))
spec["page_title"] = f"下絵の混ざりもの v5 黒い面 第{b}組 2026-11-12"
spec["decides"] = "この答えで、黒い面の目安が立った頁を対から外すかを作者と基盤が決める(何が混ざっていたかを数える)"
spec["items"] = []
for i, r in enumerate(rows):
    if (i < 17) != (b == 1):
        continue
    k = f"{r['work_id']}|{r['slug']}|{r['page']}"
    q = qc[k]
    spec["items"].append({"id": f"d{i:02d}", "version": "clip_pairs_v5", "key": k,
        "note": f"{r['work_id']} / {r['page']}",
        "images": [{"src": os.path.relpath(idx[q['sketch']], ROOT), "caption": "下絵の層(問う対象)", "target": True},
                   {"src": os.path.relpath(idx[q['line']], ROOT), "caption": "線画(見比べ用)"}]})
json.dump(spec, open(f"{D}/questions_dark34_batch{b}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(b, len(spec["items"]))
