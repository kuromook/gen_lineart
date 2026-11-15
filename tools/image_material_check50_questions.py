#!/usr/bin/env python3
"""画像素材の50枚の二択の問いを組む(登録 `image_material_50_image_check_20261114`)。

**問いと棒は画を1枚も見る前に登録済み。この器具は画面に流し込むだけで、選び方も棒も変えない。**

- 母集団: 821層(取り込んだ画像の層で、見えていて、出力に画素がある)。選び方は
  `tools/image_material_check_sample.py`(種 20261114 / 帯 5・19・26 / 1作品単位3層まで)
- 画: census 経由で抽出側が描いた 100枚(`image_material_check50_20261114/`)。
  **原稿の画素なので `data/` の外に出さない**
- 1行2枚: 仕上げの頁(赤い枠がその層の場所)と、その層だけ。問う対象は「その層だけ」の絵
- **画の限界を問いごとに書く**(納品書の逐語より。censusは確かめていない):
  1. 取り込み元の画像は道具で開けないので描かれていない(出ているのはラスターの画素だけ)
  2. 自前の合成と CSP の見た目の差が大きい8行(7・8・9・12・21・29・42・46)
  3. 枠の中の白でない画素が少ない4行(38・50・31・20)
- **位置が同じで別の頁の2行(1 と 27)は別々に聞く** —— 写しかどうかは作者が決めること
  (条例9)。異なる頁で数えると 50行 = 45頁
- 答えは JSONL に追記のみ。`pos` と直前の問いが行に入る(条例2)

**印字するもの**: 行の数 / 帯ごとの数 / 限界の札が付いた行 / 異なる頁の数 / 出力先
"""
import csv
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LIST = HERE / "data/role_predictor/image_material_check50_20261114.csv"
IMGDIR = HERE / "data/role_predictor/image_material_check50_20261114"
CHECK = HERE / "data/role_predictor/image_material_check50_check_20261115.csv"
OUT = IMGDIR / "questions_image_material50_20261115.json"

CAVEAT = ("この層の取り込み元の画像は道具で開けないので、絵には出ていません。"
          "出ているのは、この層のラスターの画素だけです。")
DIFF_HI = ("この頁の絵は、作者の目に映る頁とずれているかもしれません"
           "(トーンとグラデーションを平らに塗りました)。")
PAIR = {"1": "27", "27": "1"}


def main():
    rows = list(csv.DictReader(io.open(LIST, encoding="utf-8")))
    chk = {r["pos"]: r for r in csv.DictReader(io.open(CHECK, encoding="utf-8"))}
    assert len(rows) == 50 and len(chk) == 50, (len(rows), len(chk))

    items, tags = [], {"限界1(元の画像が出ていない)": 50, "差が大きい": [], "薄い・小さい": [], "同じ場所の別の頁": []}
    for r in rows:
        pos = r["pos"]
        c = chk[pos]
        note = [CAVEAT]
        if float(c["diff_vs_preview_255"]) > 20:
            note.append(DIFF_HI)
            tags["差が大きい"].append(pos)
        frac = float(c["layer_nonwhite_frac_in_bbox"])
        if frac < 0.02:
            note.append(f"枠の中で白でない画素は {frac*100:.2f}% です(薄い画素か、小さい画素しかありません)。")
            tags["薄い・小さい"].append(pos)
        if pos in PAIR:
            note.append(f"{PAIR[pos]} 番と、層の番号も場所も同じです。頁のファイルは別です"
                        "(同じ原稿の写しかどうかは決めていません)。この行は別に数えます。")
            tags["同じ場所の別の頁"].append(pos)
        p = f"im50_{int(pos):02d}"
        assert (IMGDIR / f"{p}_page.png").is_file() and (IMGDIR / f"{p}_layer.png").is_file(), p
        items.append({
            "id": f"im50_{int(pos):02d}",
            "version": "20261115",
            "note": " ".join(note),
            "images": [
                {"src": f"{p}_page.png", "caption": "仕上げの頁(赤い枠が、この層の場所)"},
                {"src": f"{p}_layer.png", "caption": "この層だけ(白い背景に置いたもの)", "target": True},
            ],
        })

    spec = {
        "interview_id": "image_material50_20261115",
        "page_title": "取り込んだ画像の層 50枚の二択",
        "question": "赤い枠の中のこの層は、この頁の絵の一部ですか。それとも資料(下絵・参考・見ながら描くための画像)ですか",
        "unit": "層",
        "decides": "この種別の層を機械で落としてよいかが決まります(落とす規則を据え置くか、取り消すか)",
        "initial_view": "fit",
        "choices": [
            {"value": "part_of_picture", "label": "絵の一部",
             "help": "仕上げの絵として、この頁に出ているもの(写真の背景など)"},
            {"value": "reference", "label": "資料",
             "help": "見ながら描くために置いた画像。仕上げに出すつもりのないもの"},
        ],
        "items": items,
    }
    sys.path.insert(0, str(HERE / "tools/ui"))
    from interview_spec import load_questions  # 形の点検。落ちたら書かずに止まる
    OUT.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    load_questions(OUT)

    import collections
    print("行", len(items))
    print("帯ごと", dict(collections.Counter(r["band"] for r in rows)))
    print("異なる頁", len(set(r["page_path"] for r in rows)), "/ 異なる作品単位",
          len(set(r["page_path"].split("\\")[3] for r in rows if len(r["page_path"].split("\\")) > 4)))
    for k, v in tags.items():
        print("札", k, v if isinstance(v, int) else f"{len(v)}行 {v}")
    print("出力", OUT)


main()
