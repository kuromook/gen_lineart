#!/usr/bin/env python3
"""画像素材の二択の**第2弾 (イ)** の問いを組む(登録 `image_material_second_50_check_20261115`)。

**選び方も棒も、画を見る前の登録のまま**(無作為の種 20261115、群ごと 39 / 11、1作品単位3層まで)。
置き換えは 11-15 に登録済みの3層だけ(代表でない頁)。**ここでは1行も足さず、引き直さない。**

画は第2弾の納品(`deliver_image_material_check50b_images_20261116.md`、sha256 152/152 一致)。
1行につき3枚 —— CSP が保存した頁(枠) / 原寸の合成(枠) / 層だけ(問う対象)。
**取り込んだ画像は 50行とも元の画像から描いた**(`image_fully_drawn` 50/50)。

**この便だけの事実(画面に書く)**
- **12行目は、描き直しても頁に1画素も出ない**(配置の四隅がキャンバスの外)。**外さずに出す** —— それ自体が答え
- 差がやや大きい14行に札。45行はプレビューが青みがかっている(用紙か表現色の設定。抽出側の推定)

**先に登録する点検**(答えを読む前に書く): **41行と49行は、層の画のバイトが同じ**(別のファイル・別の頁)。
**画面では知らせない。** 答えが違っても書き換えない —— 本人内一貫性の読みに使う。
46・47 は同じ頁の別の層、11・33 は画素の数が同じだけで、同じ画ではない。
"""
import collections
import csv
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LIST = HERE / "data/role_predictor/image_material_check50b_20261115.csv"
IMGDIR = HERE / "data/role_predictor/image_material_check50b_20261116"
CHECK = HERE / "data/role_predictor/image_material_check50b_check_20261116.csv"
OUT = IMGDIR / "questions_image_material50b_20261116.json"

TONE = ("原寸の合成では、トーンは網点を描かずに黒を平らに塗っています"
        "(グラデーションと色調補正も塗っていません)。CSP が保存した頁にはこの限りがありません。")
DIFF = "この行は、原寸の合成と CSP の頁で見た目の差がやや大きい行です。"
BLUE = "この頁は、CSP の頁が全体に青みがかっています(用紙か表現色の設定によるものと見ています)。"
VOID = ("この層は、描き直しても頁に1画素も出ません(置かれた位置が紙の外です)。"
        "層だけの画は白一色で、枠もありません。**そのまま出しています。**")


def main():
    rows = list(csv.DictReader(io.open(LIST, encoding="utf-8")))
    chk = {r["pos"]: r for r in csv.DictReader(io.open(CHECK, encoding="utf-8"))}
    assert len(rows) == 50 and len(chk) == 50
    assert all(c["image_fully_drawn"] == "1" for c in chk.values()), "完全に描けていない行がある"
    assert [r["pos"] for r in rows] == [str(i) for i in range(1, 51)], "並びが登録と違う"

    items, diffrows, voidrows = [], [], []
    for r in rows:
        pos = r["pos"]
        c = chk[pos]
        assert c["page_path"] == r["page_path"] and c["layer_id"] == r["layer_id"], pos
        note = [TONE]
        if float(c["diff_vs_preview_255"]) > 8:
            note.append(DIFF)
            diffrows.append(pos)
        if pos == "45":
            note.append(BLUE)
        if c["layer_px_alpha_gt0"] == "0":
            note.append(VOID)
            voidrows.append(pos)
        p = f"im50b_{int(pos):02d}"
        for suf in ("_page.png", "_page_full.png", "_layer.png"):
            assert (IMGDIR / (p + suf)).is_file(), p + suf
        items.append({
            "id": p,
            "version": "b_20261116",
            "note": " ".join(note),
            "images": [
                {"src": f"{p}_page.png", "caption": "CSP が保存した頁の見た目(赤い枠が、この層の場所)"},
                {"src": f"{p}_page_full.png", "caption": "同じ頁を原寸で重ねたもの(赤い枠が、この層の場所)"},
                {"src": f"{p}_layer.png", "caption": "この層だけ(白い背景に置いたもの)", "target": True},
            ],
        })

    spec = {
        "interview_id": "image_material50b_20261116",
        "page_title": "取り込んだ画像の層 50枚の二択(第2弾)",
        "question": "赤い枠の中のこの層は、この頁の絵の一部ですか。それとも資料(下絵・参考・見ながら描くための画像)ですか",
        "unit": "層",
        "decides": "この種別の層を機械で落としてよいかが決まります(落とす規則を据え置くか、取り消すか)",
        "initial_view": "fit",
        "choices": [
            {"value": "part_of_picture", "label": "絵の一部",
             "help": "仕上げの絵として、この頁に出ているもの(写真や3Dの背景など)"},
            {"value": "reference", "label": "資料",
             "help": "見ながら描くために置いた画像。仕上げに出すつもりのないもの"},
        ],
        "items": items,
    }
    sys.path.insert(0, str(HERE / "tools/ui"))
    from interview_spec import load_questions
    OUT.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    load_questions(OUT)

    print("行", len(items), "/ 画は1行3枚 =", len(items) * 3)
    print("群ごと", dict(collections.Counter(r["group"] for r in rows)))
    print("完全に描けた印", collections.Counter(c["image_fully_drawn"] for c in chk.values()))
    print("CSP の頁の縮尺", collections.Counter(round(float(c["preview_scale"]), 3) for c in chk.values()))
    print("札(差がやや大きい)", len(diffrows), diffrows)
    print("描き直しても画素が0", voidrows, "→ 外さずに出す(それ自体が答え)")
    print("先に登録した点検: 41 と 49 は層の画のバイトが同じ(別のファイル)。**画面では知らせない**")
    print("異なる頁のパス", len(set(r["page_path"] for r in rows)))
    print("出力", OUT)


main()
