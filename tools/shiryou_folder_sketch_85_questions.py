#!/usr/bin/env python3
"""**裁定で落ちた85層**の裁定の問いを組む(作者の求め 2026-11-17。登録 `per_layer_ruling_ui_registered_before_images_20261117`)。

**画を1枚も見る前に書いた。** 一覧 `shiryou_folder_sketch_85_20261117.csv`(68行、`a1ca583559fa7998`)の
`round` の欄どおりに分ける —— **第1便 50 / 第2便 18**(画の二択は1回50まで。作者 2026-11-10)。
`--round 1` / `--round 2` で選ぶ。**並びは一覧のまま**(種 20261117b で混ぜてある。条例2)。

**画面に書くこと**
- **68層とも、祖先のフォルダーが非表示です**(CSP が保存した頁には写りません)
- **これは『資料』のフォルダーの中に在る、名前が『下絵』『ラフ』の層です** —— 作者の形の裁定(11-16「資料」)で
  落ちている層を、**1層ずつ見てもらう**ためのものです。その形を変えるかどうかは、作者の言葉で決める
- 原寸の合成で問う層がほとんど見えない行、描き直しても画素が0 の行には札を出す((ウ) と同じ)

**ボタンは (ウ) と同じ5つ**(絵の一部 / 資料 / 下絵 / 3D背景 / 切り出し・コラージュ)+ 画面が足す
どれでもない・分からない。**並びは変えない。**
"""
import argparse
import collections
import csv
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LIST = HERE / "data/role_predictor/shiryou_folder_sketch_85_20261117.csv"
IMGDIR = HERE / "data/role_predictor/shiryou_folder_sketch_im85_20261117"
CHECK = IMGDIR.parent / "shiryou_folder_sketch_im85_check_20261117.csv"
FRAME = IMGDIR.parent / "shiryou_folder_sketch_im85_frame_check_20261117.csv"
CHOICES = HERE / "config/image_material_choices_20261116.json"

HIDDEN = ("この層は、**親のフォルダーが非表示**です。CSP が保存した頁には写っていません —— "
          "赤い枠が、その層の在った場所です。")
WHERE = ("この層は、**名前に『下絵』や『ラフ』があり、『資料』のフォルダーの中に在る**層です。")
TONE = ("原寸の合成では、トーンは網点を描かずに黒を平らに塗っています"
        "(グラデーションと色調補正も塗っていません)。")
FAINT = ("原寸の合成では、この層は**ほとんど見えていません**"
         "(親フォルダーの不透明度やマスク、上の層に隠れているため)。**下の『この層だけ』の画で判じてください。**")
VOID = ("この層は、描き直しても頁に1画素も出ません。層だけの画は白一色で、枠もありません。**そのまま出しています。**")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=(1, 2), required=True)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(io.open(LIST, encoding="utf-8"))
            if int(r["round"]) == a.round]
    chk = {r["pos"]: r for r in csv.DictReader(io.open(CHECK, encoding="utf-8"))}
    frm = {r["pos"]: r for r in csv.DictReader(io.open(FRAME, encoding="utf-8"))}
    assert len(chk) == 68 and len(frm) == 68, (len(chk), len(frm))
    assert len(rows) == (50 if a.round == 1 else 18), len(rows)
    assert all(r["hidden_by"] == "ancestor" for r in rows), "祖先が非表示でない行がある"

    items, faint, voids = [], [], []
    for r in rows:
        pos, c = r["pos"], chk[r["pos"]]
        assert c["page_path"] == r["page_path"] and c["layer_id"] == r["layer_id"], pos
        note = [HIDDEN, WHERE, TONE]
        if c.get("image_fully_drawn") not in (None, "", "1"):
            note.append("**この層は、元の画像を全部読めていません**(抽出側の点検)。")
        if frm[pos].get("asked_shown_frac") and float(frm[pos]["asked_shown_frac"]) < 0.5:
            note.append(FAINT)
            faint.append(pos)
        if c.get("layer_px_alpha_gt0") == "0":
            note.append(VOID)
            voids.append(pos)
        p = f"im85_{int(pos):02d}"
        for suf in ("_page.png", "_page_full.png", "_layer.png"):
            assert (IMGDIR / (p + suf)).is_file(), p + suf
        items.append({
            "id": p,
            "version": f"im85_r{a.round}_20261117",
            "note": " ".join(note),
            "images": [
                {"src": f"{p}_page.png", "caption": "CSP が保存した頁の見た目(赤い枠が、この層の在った場所。層は非表示なので写りません)"},
                {"src": f"{p}_page_full.png", "caption": "同じ頁を原寸で重ねたもの(この層だけは非表示でも描いてあります)"},
                {"src": f"{p}_layer.png", "caption": "この層だけ(白い背景に置いたもの)", "target": True},
            ],
        })

    choices = json.loads(CHOICES.read_text(encoding="utf-8"))["choices"]
    assert [c["label"] for c in choices] == [
        "絵の一部", "資料", "下絵", "3D背景", "切り出し・コラージュ"], "ボタンの並びが登録と違う"
    spec = {
        "interview_id": f"im85_r{a.round}_20261117",
        "page_title": f"資料のフォルダーの中の『下絵』『ラフ』の層({a.round}回目 / 全2回)",
        "question": "赤い枠の中のこの層は、何ですか",
        "unit": "層",
        "decides": ("この層を機械で落としてよいかが決まります"
                    "(『資料のフォルダーの中の下絵は資料』という形の裁定を、1層ずつ確かめます)"),
        "initial_view": "fit",
        "choices": choices,
        "items": items,
    }
    out = IMGDIR / f"questions_im85_r{a.round}_20261117.json"
    sys.path.insert(0, str(HERE / "tools/ui"))
    from interview_spec import load_questions
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    load_questions(out)

    print(f"{a.round}回目: 行 {len(items)} / 画 {len(items) * 3}")
    print("ボタン", [c["label"] for c in choices], "+ 画面が足す(どれでもない・分からない)")
    print("作品単位", dict(collections.Counter(r["split_unit"].rsplit("\\", 1)[-1] for r in rows)))
    print("年の札", dict(collections.Counter(r["year_label"] for r in rows)))
    print("覆う写しの層", sum(int(r["covers_copies"]) for r in rows))
    print("札(原寸でほとんど見えない)", len(faint), faint)
    print("描き直しても画素が0", voids or "0", "→ 外さずに出す")
    print("出力", out)


main()
