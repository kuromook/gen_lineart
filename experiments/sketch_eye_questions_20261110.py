"""Questions JSON for the author's eye check of v4 sketches (2026-11-10). Uses foundation's tools/ui (server, --lan).
One item per page in the measured population (1,349, load_jobs order = work order). Images are the delivered full pages
(not downscaled; the UI scales). Output under data/ (manuscript names; not tracked)."""
import json, os, sys
sys.path.insert(0, "tools/evaluation")
from pair_feature_tiles import load_jobs
ROOT = "data/clip_pairs_v4/clip_pairs"
jobs = load_jobs("data/clip_pairs_v4")
spec = {
    "interview_id": "pair_signal_sketch_clean_20261110",
    "page_title": "下絵の混ざりもの 2026-11-10",
    "question": "左の下絵の層に、線画にならない物が混ざっていますか?",
    "unit": "頁の下絵の層",
    "decides": "この答えで、取り直す集計からこの頁を外すか(何が混ざっていたかを数える)が決まる",
    "choices": [
        {"value": "clean", "label": "混ざっていない", "help": "下絵の線だけ(線画に無い描き込みがあっても下絵なら可)"},
        {"value": "text", "label": "ネームの文字・書き込み", "help": "台詞の下書き・指示・メモ"},
        {"value": "frame", "label": "コマ枠・定規線", "help": "下絵に引いたコマ枠・パースの補助線"},
        {"value": "tone", "label": "灰色の塗り・トーン", "help": "灰色の面・影の当たり・塗りつぶし"},
        {"value": "other_picture", "label": "別の絵・資料・版違い", "help": "線画と構図が違う・資料・別頁の絵"},
        {"value": "several", "label": "2つ以上ある", "help": "どれが混ざっているかを一言欄に書く"},
    ],
    "items": [],
}
for n, (key, sp, lp) in enumerate(jobs):
    w, slug, page = key.split("|")
    spec["items"].append({
        "id": f"p{n:04d}", "version": "clip_pairs_v4",
        "note": f"{w} / {page}",
        "images": [{"src": os.path.relpath(sp, ROOT), "caption": "下絵の層(問う対象)", "target": True},
                   {"src": os.path.relpath(lp, ROOT), "caption": "線画(見比べ用)"}],
        "key": key,
    })
json.dump(spec, open("data/sketch_eye_check_20261110/questions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(spec["items"]), "items")
