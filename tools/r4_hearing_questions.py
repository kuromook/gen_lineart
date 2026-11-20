# -*- coding: utf-8 -*-
"""R4 の聞き取り(30層、段の4択)の問いの JSON を、census の聞き取りの表と抽出側の画から作る。

登録: config/prereg_r4_film_rule_nashi_20261120.md(組は伏せる・層名は隠す・表の行の順 = 混ぜた順)
入力: data/census/r4_film_nashi_hearing_table_20261120.csv / r4_film_nashi_images_20261120/
出力: data/census/r4_film_nashi_images_20261120/questions.json(原稿の名前を含むので追跡しない)
鍵(組)は読まない。使い方: ./venv/bin/python tools/r4_hearing_questions.py
"""
import csv, io, json, sys

sys.path.insert(0, "tools/ui")
from interview_spec import load_questions  # noqa: E402

D = "data/census"
IMG = D + "/r4_film_nashi_images_20261120"
TABLE = D + "/r4_film_nashi_hearing_table_20261120.csv"
CHECK = D + "/r4_film_nashi_images_check_20261120.csv"
FRAME = D + "/r4_film_nashi_images_frame_check_20261120.csv"


def main():
    rows = list(csv.DictReader(io.open(TABLE, encoding="utf-8")))
    chk = {r["pos"]: r for r in csv.DictReader(io.open(CHECK, encoding="utf-8"))}
    frm = {r["pos"]: r for r in csv.DictReader(io.open(FRAME, encoding="utf-8"))}
    items = []
    for r in rows:
        p = r["row"]; c = chk[p]
        assert (c["page_path"], c["layer_id"]) == (r["page_path"], r["layer_id"]), p
        hidden = not (r["self_shown"] == "1" and r["ancestors_shown"] == "1")
        notes = []
        if hidden:
            notes.append("この層は CSP では非表示です(本番の頁には出ていません)。原寸の頁では、この層だけ表示にして描いています")
        if frm[p]["layer_nonwhite_px"] == "0":
            notes.append("この層は白一色で描かれているので、層だけの画は真っ白に見えます。原寸の頁で白が乗っている所を見てください")
        n = "%02d" % int(p)
        items.append({
            "id": "r4_%s" % n,
            "note": "。".join(notes),
            "images": [
                {"src": "imr4_%s_page.png" % n, "caption": "頁(赤枠 = 層の範囲)"},
                {"src": "imr4_%s_page_full.png" % n, "caption": "原寸の頁(この層を含めて描いた)"},
                {"src": "imr4_%s_layer.png" % n, "caption": "この層だけ(白に重ねた)", "target": True},
            ],
        })
    spec = {
        "interview_id": "r4_film_nashi_stage_20261120",
        "page_title": "R4 層の段 30",
        "question": "この層は、描く工程のどの段ですか?",
        "unit": "層(印の付いた1枚)",
        "decides": "名前で段の決まらない層を、画の性質で あたり・下絵 と言えるかを確かめる",
        "choices": [
            {"value": "atari", "label": "あたり", "help": "形や配置をとる、ごく粗い描き"},
            {"value": "shitae", "label": "下絵", "help": "清書の前の下描き"},
            {"value": "line", "label": "線画(清書)", "help": "仕上げの線"},
        ],
        "items": items,
    }
    out = IMG + "/questions.json"
    json.dump(spec, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    s = load_questions(out)
    print("ok", out, len(s["items"]), "items;",
          "hidden", sum("非表示" in i["note"] for i in items), "white", sum("白一色" in i["note"] for i in items))


if __name__ == "__main__":
    main()
