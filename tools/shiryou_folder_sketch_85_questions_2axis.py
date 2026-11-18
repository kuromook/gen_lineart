#!/usr/bin/env python3
"""**85層の裁定を2つの軸で問い直す**(作者の求め 2026-11-18)。登録 `two_axis_ruling_registered_before_answers_20261118`。

**作者の言(逐語)**「そうだな、つまり２つボタンを押せるようにすればいいのだ 下絵（のスクリーンショット）であり資料 、とか
線画（のスクリーンショット）であり切り出し、というふうに これならレイヤの発注時には機械的に除外しつつ、
そのレイヤが持つ特徴量を参考にすることができる」

**軸1 = 用途**(落とすかを決める。a はこの列だけ読む)/ **軸2 = 見た目**(特徴量の参考。落とすのに使わない)。
表は `config/two_axis_choices_20261118.json`(並びは登録のまま assert する)。

**便の分け方**
- `--round 1b`: 第1便の50層。**用途は作者が 11-18 に押した答え(50層とも『資料』)を初期値に置き**、
  **見た目の列を空で聞く**。初期値は**変えられる**(閉じない)。変えた数は読む器具が印字する
- `--round 2`: 第2便の18層。**両方空で聞く**(用途を一度も聞いていない層)

**便で文脈が違うことを隠さない**(条例2): 1b は用途が埋まった状態で、2 は空で始まる。
**これは裁定で、棒は無い**(規律9-5)—— (ウ) の 0/50 も名前の規則も a の数も動かない。
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
R1ANS = IMGDIR.parent / "shiryou_folder_sketch_85_answers_r1_20261117.jsonl"
CHOICES = HERE / "config/two_axis_choices_20261118.json"

HIDDEN = ("この層は、**親のフォルダーが非表示**です。CSP が保存した頁には写っていません —— "
          "赤い枠が、その層の在った場所です。")
WHERE = "この層は、**名前に『下絵』や『ラフ』があり、『資料』のフォルダーの中に在る**層です。"
TONE = ("原寸の合成では、トーンは網点を描かずに黒を平らに塗っています"
        "(グラデーションと色調補正も塗っていません)。")
FAINT = ("原寸の合成では、この層は**ほとんど見えていません**"
         "(親フォルダーの不透明度やマスク、上の層に隠れているため)。**下の『この層だけ』の画で判じてください。**")
PREFILLED = ("**上の列(用途)には、あなたが前の便で押した答えを置いてあります。** 変えてかまいません。"
             "下の列(見た目)は空です。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", choices=("1b", "2"), required=True)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(io.open(LIST, encoding="utf-8"))
            if r["round"] == ("1" if a.round == "1b" else "2")]
    chk = {r["pos"]: r for r in csv.DictReader(io.open(CHECK, encoding="utf-8"))}
    frm = {r["pos"]: r for r in csv.DictReader(io.open(FRAME, encoding="utf-8"))}
    assert len(chk) == 68 and len(frm) == 68, (len(chk), len(frm))
    assert len(rows) == (50 if a.round == "1b" else 18), len(rows)
    assert all(r["hidden_by"] == "ancestor" for r in rows), "祖先が非表示でない行がある"

    cfg = json.loads(CHOICES.read_text(encoding="utf-8"))
    ax1, ax2 = cfg["axis1"]["choices"], cfg["axis2"]["choices"]
    # **並びは登録のまま**(条例2)。11-16 の5つから 3D背景 を軸2へ移し、残り4つの相対の並びは変えていない
    assert [c["label"] for c in ax1] == ["絵の一部", "資料", "下絵として使う", "切り出し・コラージュ"], "軸1 の並びが登録と違う"
    assert [c["label"] for c in ax2] == ["ラフ・下絵", "線画", "彩色された絵", "写真・取り込んだ画像",
                                         "3D", "塗り・トーン"], "軸2 の並びが登録と違う"

    # 第1便の答え(用途)を初期値にする。**層ごとに最後の行を採る。無い層には置かない**
    prev = {}
    if a.round == "1b":
        assert R1ANS.is_file(), R1ANS
        for line in io.open(R1ANS, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                if rec.get("item_id"):
                    prev[rec["item_id"]] = rec
        assert len(prev) == 50, len(prev)
        vals = {c["value"] for c in ax1}
        for iid, rec in prev.items():
            assert rec["value"] in vals, (iid, rec["value"], "軸1 に無い答えは初期値に置けない")

    items, faint, filled = [], [], collections.Counter()
    for r in rows:
        pos, c = r["pos"], chk[r["pos"]]
        assert c["page_path"] == r["page_path"] and c["layer_id"] == r["layer_id"], pos
        p = f"im85_{int(pos):02d}"
        note = [HIDDEN, WHERE, TONE]
        if frm[pos].get("asked_shown_frac") and float(frm[pos]["asked_shown_frac"]) < 0.5:
            note.append(FAINT)
            faint.append(pos)
        it = {
            "id": p,
            "version": f"im85_{a.round}_2axis_20261118",
            "images": [
                {"src": f"{p}_page.png", "caption": "CSP が保存した頁の見た目(赤い枠が、この層の在った場所。層は非表示なので写りません)"},
                {"src": f"{p}_page_full.png", "caption": "同じ頁を原寸で重ねたもの(この層だけは非表示でも描いてあります)"},
                {"src": f"{p}_layer.png", "caption": "この層だけ(白い背景に置いたもの)", "target": True},
            ],
        }
        if p in prev:
            it["prefill"] = prev[p]["value"]      # **用途だけ。見た目には初期値を置かない**
            filled[prev[p]["label"]] += 1
            note.append(PREFILLED)
        it["note"] = " ".join(note)
        for suf in ("_page.png", "_page_full.png", "_layer.png"):
            assert (IMGDIR / (p + suf)).is_file(), p + suf
        items.append(it)

    spec = {
        "interview_id": f"im85_{a.round}_2axis_20261118",
        "page_title": f"資料のフォルダーの中の『下絵』『ラフ』の層 —— 用途と見た目の2つを押す({a.round})",
        "question": cfg["axis1"]["question"],
        "question2": cfg["axis2"]["question"],
        "unit": "層",
        "decides": ("上の列(用途)で、この層を機械で落としてよいかが決まります。"
                    "下の列(見た目)は落とす・落とさないには使わず、層の特徴を見るために残します"),
        "initial_view": "fit",
        "choices": ax1,
        "choices2": ax2,
        "items": items,
    }
    out = IMGDIR / f"questions_im85_{a.round}_2axis_20261118.json"
    sys.path.insert(0, str(HERE / "tools/ui"))
    from interview_spec import load_questions
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    load_questions(out)

    print(f"便 {a.round}: 行 {len(items)} / 画 {len(items) * 3}")
    print("軸1(用途。a が読む) ", [c["label"] for c in ax1], "+ 画面が足す(どれでもない・分からない)")
    print("軸2(見た目。落とすのに使わない)", [c["label"] for c in ax2], "+ 同じ2つ")
    print("用途の初期値を置いた層", sum(filled.values()), dict(filled) or "0(両方空で聞く)")
    print("見た目の初期値", "置いていない(作者が押していない答えは置かない)")
    print("作品単位", dict(collections.Counter(r["split_unit"].rsplit("\\", 1)[-1] for r in rows)))
    print("覆う写しの層", sum(int(r["covers_copies"]) for r in rows))
    print("札(原寸でほとんど見えない)", len(faint), faint)
    print("出力", out)


main()
