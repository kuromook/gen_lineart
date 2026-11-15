#!/usr/bin/env python3
"""画像素材の50枚の二択・第2版の問いを組む(**仕切り直し**。登録 `image_material_50_image_check_20261114`)。

**選び方も棒も 11-14 の登録のまま。再抽選しない**(同じ50行)。第1版の11行の答えは版として凍結してあり、
棒は**この第2版の50行だけ**で読む。

第1版との違いは**画だけ**:
- 取り込んだ画像を**元の画像(`ResizableOriginalMipmap`)から描き直した** —— 50行とも「完全に描けた」の印が立った
  (読めなかった塊 0)。第1版は描画用のミップマップの**古い切れ端**を描いていた(抽出側の実測)
- **CSP が保存した頁の見た目(`CanvasPreview`)**が付いた。縮尺は行ごとに 1/4 か 1/8
- 枠は、一覧の bbox ではなく**描き直した層の画素の bbox** から引いた(一覧とは位置が違う。重なり 0 の行が11)
- 1行につき**3枚**: CSP の頁(枠) / 原寸の合成(枠。層の画と同じ大きさ) / 層だけ(問う対象)

残る限界(問いごとに画面に書く): **原寸の合成では、トーンは網点を描かず黒を平らに塗り、グラデーションと
色調補正は塗っていない**(CSP の頁にはこの限界が無い)。差がやや大きい10行には札を立てる。

**1行目と27行目は、同じ中身のファイルの同じ層**(census が sha256 で確かめた)。**画面では知らせない** ——
**本人内一貫性の点検**として使う(先に登録した。答えが違っても書き換えない)。

**印字するもの**: 行の数 / 帯ごと / 札の付いた行 / 同じ層の組 / 出力先。
"""
import csv
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LIST = HERE / "data/role_predictor/image_material_check50_20261114.csv"
IMGDIR = HERE / "data/role_predictor/image_material_check50_20261115"
CHECK = HERE / "data/role_predictor/image_material_check50_check_20261115_v2.csv"
OUT = IMGDIR / "questions_image_material50_v2_20261115.json"

TONE = ("原寸の合成では、トーンは網点を描かずに黒を平らに塗っています"
        "(グラデーションと色調補正も塗っていません)。CSP が保存した頁にはこの限りがありません。")
DIFF = "この頁は、原寸の合成と CSP の頁の差がやや大きい行です(頁にトーンの層があります)。"


def main():
    rows = list(csv.DictReader(io.open(LIST, encoding="utf-8")))
    chk = {r["pos"]: r for r in csv.DictReader(io.open(CHECK, encoding="utf-8"))}
    assert len(rows) == 50 and len(chk) == 50
    assert all(c["image_fully_drawn"] == "1" for c in chk.values()), "完全に描けていない行がある"

    items, diffrows = [], []
    for r in rows:
        pos = r["pos"]
        c = chk[pos]
        note = [TONE]
        if float(c["diff_vs_preview_255"]) > 8:
            note.append(DIFF)
            diffrows.append(pos)
        p = f"im50_{int(pos):02d}"
        for suf in ("_page.png", "_page_full.png", "_layer.png"):
            assert (IMGDIR / (p + suf)).is_file(), p + suf
        items.append({
            "id": f"im50_{int(pos):02d}",
            "version": "v2_20261115",
            "note": " ".join(note),
            "images": [
                {"src": f"{p}_page.png", "caption": "CSP が保存した頁の見た目(赤い枠が、この層の場所)"},
                {"src": f"{p}_page_full.png", "caption": "同じ頁を原寸で重ねたもの(赤い枠が、この層の場所)"},
                {"src": f"{p}_layer.png", "caption": "この層だけ(白い背景に置いたもの)", "target": True},
            ],
        })

    spec = {
        "interview_id": "image_material50_v2_20261115",
        "page_title": "取り込んだ画像の層 50枚の二択(描き直した画)",
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
    from interview_spec import load_questions
    OUT.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    load_questions(OUT)

    import collections
    print("行", len(items), "/ 画は1行3枚 =", len(items) * 3)
    print("帯ごと", dict(collections.Counter(r["band"] for r in rows)))
    print("完全に描けた印", collections.Counter(c["image_fully_drawn"] for c in chk.values()))
    print("CSP の頁の縮尺", collections.Counter(round(float(c["preview_scale"]), 3) for c in chk.values()))
    print("札(差がやや大きい)", len(diffrows), diffrows)
    print("同じ中身のファイルの組: 1 と 27(同じ層。**画面では知らせない** = 本人内一貫性の点検) / 3 と 32(層は違う)")
    print("異なる頁のパス", len(set(r["page_path"] for r in rows)))
    print("出力", OUT)


main()
