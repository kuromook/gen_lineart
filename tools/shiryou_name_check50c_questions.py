#!/usr/bin/env python3
"""(ウ) **資料の名前の規則の検定**の問いを組む(登録 `validation_hearing_registered_before_images`)。

**選び方も棒も、画を見る前の登録のまま** —— 種 20261117、群ごと 7 / 43、1作品単位3層まで、
代表でない頁の層は census の印(鍵 v3)で飛ばして**不動点**(一覧 v4 `bb3bdfd3aad798a2`)。
**ここでは1行も足さず、引き直さない。**

**この便の要点(画面に書く)**
- **50層とも非表示です**(層が非表示 1 / 祖先が非表示 40 / 両方 9)。**CSP が保存した頁には写りません** ——
  枠だけが、その層の在った場所です。原寸の合成では、**その層だけ**を非表示でも描いてもらいました
- **ボタンは5つ**(作者の承認「2 ok」、2026-11-16): 絵の一部 / 資料 / 下絵 / 3D背景 / 切り出し・コラージュ。
  並びは (イ) の並びに挟んだまま(並びを変えると押し方が動きうる。条例2)。「どれでもない」「分からない」は画面が足す
- 描き直しても画素が 0 の層は**外さずに出す**((イ) と同じ。それ自体が答え)

**棒**(答えを読む前の登録。`shiryou_name_check50c_read_bar.py` が読む): 「資料」でない答え
(絵の一部 + 下絵 + 3D背景 + 切り出し・コラージュ)が **0〜2 → 名前の規則を据え置き** /
**3以上 → 撤回(215,247 → 215,961 相当)** / **「どれでもない」+「分からない」が10以上 → 判定保留**。
"""
import collections
import csv
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LIST = HERE / "data/role_predictor/shiryou_name_check50_v4_20261117.csv"
IMGDIR = HERE / "data/role_predictor/shiryou_name_check50c_20261117"
CHECK = IMGDIR.parent / "shiryou_name_check50c_check_20261117.csv"
OUT = IMGDIR / "questions_shiryou_name50c_20261117.json"
CHOICES = HERE / "config/image_material_choices_20261116.json"

HIDDEN = ("この層は、CSP では非表示です(層そのものか、親のフォルダーが非表示)。"
          "**CSP が保存した頁には写っていません** —— 赤い枠が、その層の在った場所です。")
TONE = ("原寸の合成では、トーンは網点を描かずに黒を平らに塗っています"
        "(グラデーションと色調補正も塗っていません)。CSP が保存した頁にはこの限りがありません。")
DIFF = "この行は、原寸の合成と CSP の頁で見た目の差がやや大きい行です。"
VOID = ("この層は、描き直しても頁に1画素も出ません。"
        "層だけの画は白一色で、枠もありません。**そのまま出しています。**")


def main():
    rows = list(csv.DictReader(io.open(LIST, encoding="utf-8")))
    chk = {r["pos"]: r for r in csv.DictReader(io.open(CHECK, encoding="utf-8"))}
    assert len(rows) == 50 and len(chk) == 50, (len(rows), len(chk))
    assert [r["pos"] for r in rows] == [str(i) for i in range(1, 51)], "並びが登録と違う"
    assert all(r["hidden_by"] in ("self", "ancestor", "both") for r in rows), "非表示でない行がある"

    items, diffrows, voidrows, partial = [], [], [], []
    for r in rows:
        pos, c = r["pos"], chk[r["pos"]]
        assert c["page_path"] == r["page_path"] and c["layer_id"] == r["layer_id"], pos
        note = [HIDDEN, TONE]
        if c.get("image_fully_drawn") not in (None, "", "1"):
            partial.append(pos)
            note.append("**この層は、元の画像を全部読めていません**(抽出側の点検)。")
        if c.get("diff_vs_preview_255") and float(c["diff_vs_preview_255"]) > 8:
            note.append(DIFF)
            diffrows.append(pos)
        if c.get("layer_px_alpha_gt0") == "0":
            note.append(VOID)
            voidrows.append(pos)
        p = f"im50c_{int(pos):02d}"
        for suf in ("_page.png", "_page_full.png", "_layer.png"):
            assert (IMGDIR / (p + suf)).is_file(), p + suf
        items.append({
            "id": p,
            "version": "c_20261117",
            "note": " ".join(note),
            "images": [
                {"src": f"{p}_page.png", "caption": "CSP が保存した頁の見た目(赤い枠が、この層の在った場所。層は非表示なので写りません)"},
                {"src": f"{p}_page_full.png", "caption": "同じ頁を原寸で重ねたもの(この層だけは非表示でも描いてあります)"},
                {"src": f"{p}_layer.png", "caption": "この層だけ(白い背景に置いたもの)", "target": True},
            ],
        })

    choices = json.loads(CHOICES.read_text(encoding="utf-8"))["choices"]
    assert len(choices) == 5 and [c["label"] for c in choices] == [
        "絵の一部", "資料", "下絵", "3D背景", "切り出し・コラージュ"], "ボタンの並びが登録と違う"
    spec = {
        "interview_id": "shiryou_name50c_20261117",
        "page_title": "非表示の取り込んだ画像の層 50枚(名前の規則の検定)",
        "question": "赤い枠の中のこの層は、何ですか",
        "unit": "層",
        "decides": ("名前に『資料』の語がある非表示の層 3,557 を機械で落としてよいかが決まります"
                    "(落とす規則を据え置くか、取り消して候補に戻すか)"),
        "initial_view": "fit",
        "choices": choices,
        "items": items,
    }
    sys.path.insert(0, str(HERE / "tools/ui"))
    from interview_spec import load_questions
    OUT.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    load_questions(OUT)

    print("行", len(items), "/ 画は1行3枚 =", len(items) * 3)
    print("ボタン", [c["label"] for c in choices], "+ 画面が足す(どれでもない・分からない)")
    print("群ごと", dict(collections.Counter(r["group"] for r in rows)))
    print("非表示の形", dict(collections.Counter(r["hidden_by"] for r in rows)))
    print("年の札", dict(collections.Counter(r["year_label"] for r in rows)))
    print("完全に描けていない行", partial or "0")
    print("札(差がやや大きい)", len(diffrows), diffrows)
    print("描き直しても画素が0", voidrows or "0", "→ 外さずに出す(それ自体が答え)")
    print("異なる頁のパス", len(set(r["page_path"] for r in rows)))
    print("出力", OUT)


main()
