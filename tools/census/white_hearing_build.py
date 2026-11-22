#!/usr/bin/env python
"""**白い層の例外の聞き取り(20層)の問いの JSON**(Track E の一覧と問いの案、2026-11-22。ユーザー「画面を作る(今回は例外)」11-23)。

- 行 = E の一覧 `outbox/white_exception_hearing_list_20261122.csv` の 20層。画は抽出側の便(80枚、sha 81本一致)。census は画を開かない(ファイル名で繋ぐだけ)
- 問いは E の案のまま。**ボタンは出現数に合わせる**: E の案の4つのうち「3D の線画なので別の区分」は E の読みで 1層なので置かず、一言の欄に回す
  (「どれでもない」「分からない」と一言は部品が足す)
- **E の区分・層の段(前の答え)・層の群(stratum)は画面に出さない**(答えを寄せる)。**層名は画面にだけ出す**(E の同意。一覧・通達・公開物には書かない)
- 出す順は種 20261123 で混ぜる(群が続けて出ないように)。出した位置は答えの行に残る
- 抽出側の照合で「消しても切り出しの中が変わらない」層には、その事実を一言添える(理由は書かない。調べていない)

出力 `data/census_full/white_hearing/questions.json`(非公開)。
"""
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIST = os.path.join(ROOT, "outbox", "white_exception_hearing_list_20261122.csv")
CHK = os.path.join(ROOT, "data", "census_full", "white_exception_hearing_images_check_20261123.csv")
OUT = os.path.join(ROOT, "data", "census_full", "white_hearing", "questions.json")
SEED = 20261123


def main() -> int:
    L = list(csv.DictReader(open(LIST, encoding="utf-8")))
    K = {r["pos"]: r for r in csv.DictReader(open(CHK, encoding="utf-8"))}
    assert len(L) == 20 and all((K[r["pos"]]["page_path"], K[r["pos"]]["layer_id"]) == (r["page_path"], r["layer_id"]) for r in L)
    c = db.connect()
    items = []
    for r in L:
        k = K[r["pos"]]
        nm = c.execute("select coalesce(name,'') from layers where page_path = ? and layer_id = ?",
                       [r["page_path"], int(r["layer_id"])]).fetchone()[0]
        note = [f"層の名前: {nm or '(名前なし)'}"]
        if k["layer_shown_bit0"] == "0":
            note.append("この層は原稿で非表示です。")
        if k["without_same_as_page_in_crop"] == "1":
            note.append("この層を消しても、切り出した範囲の見た目は変わりません(④ = 頁の同じ範囲)。")
        if k["note_page"]:
            note.append("頁の絵はクリスタの見た目と同じではありません(トーンは平らな灰、グラデーションは塗らない等)。")
        n = f"{int(r['pos']):02d}"
        items.append({"id": f"wh_{n}", "version": "20261123", "note": " ".join(note), "images": [
            {"src": k["file_page"], "caption": "① 頁の絵(縮小)。赤い枠が この層の範囲", "lead": True},
            {"src": k["file_grey"], "caption": "② この層だけ(灰色の地・原寸)", "target": True},
            {"src": k["file_checker"], "caption": "③ この層だけ(市松の地・原寸)", "target": True},
            {"src": k["file_without"], "caption": "④ この層を消した頁(②と同じ範囲・原寸)"}]})
    random.Random(SEED).shuffle(items)
    spec = {
        "interview_id": "census_white_exception_20261123", "page_title": "白い層の聞き取り",
        "question": "この層は「白い塗りの層」として、絵から除外してよいですか?(この頁だけでなく、どの頁でも同じ扱いにしてよいか)",
        "unit": "層",
        "decides": "白い塗りの層を名前や白さで自動で外す決まりを、どこまで使ってよいかが決まる",
        "choices": [
            {"value": "exclude_ok", "label": "除外してよい(白い塗り)", "help": "ホワイト・白消しなど。絵の線や面ではない"},
            {"value": "keep", "label": "除外してはいけない(絵の一部)", "help": "白くても絵として要る層"},
            {"value": "balloon_lettering", "label": "フキダシ・描き文字なので別の区分", "help": "白い面はフキダシや描き文字のもの"},
        ],
        "items": items,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(spec, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"問い {len(items)} / 画 {sum(len(i['images']) for i in items)} / 出す順の先頭 {[i['id'] for i in items[:5]]}")
    print(f"出力: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
