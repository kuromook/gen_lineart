#!/usr/bin/env python3
"""**3D の疑いの『線画』33層の裁定の問いを組む** —— 画は届いているが、**答えを1つも見る前に書いた**。
登録 `suspect33_3d_ruling_registered_before_answers_20261119`。

**作者の指示(2026-11-18、逐語)**「３Dレイヤの裁定はこちらでやる」—— 画面は E が作り、答えは census に返す。
**基盤の通達(ユーザー、2026-11-18、逐語)**「OK 3Dについては名前で引けないものの自動判定は難しく
それらは裁定で決めることとした」。

**a はこの便の答えを読まない**(名前で引けない 3D は a の外)。**a の数は動かない**。
ボタンの表は `config/suspect33_3d_choices_20261119.json`(並びは assert する。条例2)。
**`--one-axis` で軸2(何を描いたか)を落とせる**(作者が要らないと言ったとき)。

祖先のフォルダー名は census の一覧が届いていれば**層ごとに**出し、無ければ**母集団の言い方**にする
(**E は原稿の列を自分で作らない**)。
"""
import argparse
import collections
import csv
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
D = HERE / "data/role_predictor"
IMGDIR = D / "succor_3d_suspect33_images_20261118"
CHECK = D / "succor_3d_suspect33_check_20261118.csv"
FRAME = D / "succor_3d_suspect33_frame_check_20261118.csv"
ORDER = [HERE / "inbox/succor_3d_suspect33_images_order_20261118.csv",
         D / "succor_3d_suspect33_images_order_20261118.csv"]
CHOICES = HERE / "config/suspect33_3d_choices_20261119.json"

SHOWN = "**この層は CSP で見えています**(33層とも表示中です)。赤い枠が、この層の在る場所です。"
NAME = "**この層の名前は『線画』です** —— 名前では 3D か人の線か引けないので、1層ずつ見てもらっています。"
TONE = ("原寸の合成では、トーンは網点を描かずに黒を平らに塗っています"
        "(グラデーションと色調補正も塗っていません)。")
FAINT = ("原寸の頁では、この層は**ほとんど見えていません**"
         "(半透明・クリッピング・コマのマスクのため)。**下の『この層だけ』の画で判じてください。**")
CLIP = "**この層はクリッピングです。** 基の層を描かずに、切らずに塗ってあります(抽出側の注記)。"
SPREAD = "**この頁は見開きです**(とても横長の画です)。"
POP = "33層とも、背景・小物のフォルダーの中に在ります(層ごとの名前は届いていません)。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-axis", action="store_true", help="軸2(何を描いたか)を出さない")
    a = ap.parse_args()

    chk = list(csv.DictReader(io.open(CHECK, encoding="utf-8")))
    frm = {r["pos"]: r for r in csv.DictReader(io.open(FRAME, encoding="utf-8"))}
    assert len(chk) == 33 and len(frm) == 33, (len(chk), len(frm))
    assert all(r["layer_visible_in_csp"] == "1" for r in chk), "CSP で見えない層がある(便の前提と違う)"

    order = {}
    for p in ORDER:
        if p.is_file():
            order = {r["pos"]: r for r in csv.DictReader(io.open(p, encoding="utf-8"))}
            print(f"census の一覧を使う: {p.name}(33行)" if len(order) == 33 else f"一覧の行が 33 でない: {len(order)}")
            break
    if not order:
        print("**census の一覧がまだ無い** → 祖先は母集団の言い方で出す(E は原稿の列を自分で作らない)")

    cfg = json.loads(CHOICES.read_text(encoding="utf-8"))
    ax1 = cfg["axis1"]["choices"]
    ax2 = cfg["axis2"]["choices"]
    assert [c["label"] for c in ax1] == ["3D から出した線", "3D をなぞって描いた線", "人が描いた線"], "軸1 の並びが登録と違う"
    assert [c["label"] for c in ax2] == ["背景(建物・部屋・風景)", "小物・道具", "人物", "効果・模様"], "軸2 の並びが登録と違う"

    items, faint, clips, spreads = [], [], [], []
    for r in chk:
        pos = r["pos"]
        p = f"d3s_{int(pos):02d}"
        note = [SHOWN, NAME]
        if order:
            anc = (order[pos].get("ancestors") or "").strip()
            assert order[pos]["page_path"] == r["page_path"] and order[pos]["layer_id"] == r["layer_id"], pos
            if anc:
                note.append(f"この層は、**{anc}** のフォルダーの中に在ります。")
        else:
            note.append(POP)
        note.append(TONE)
        if int(r["canvas_w"]) > 10000:
            note.append(SPREAD)
            spreads.append(pos)
        if "クリッピング" in (r.get("not_painted") or "") or "クリッピング" in (r.get("note") or ""):
            note.append(CLIP)
            clips.append(pos)
        if frm[pos].get("asked_shown_frac") and float(frm[pos]["asked_shown_frac"]) < 0.5:
            note.append(FAINT)
            faint.append(pos)
        for suf in ("_page.png", "_page_full.png", "_layer.png"):
            assert (IMGDIR / (p + suf)).is_file(), p + suf
        items.append({
            "id": p,
            "version": "d3s_20261119" + ("_1axis" if a.one_axis else ""),
            "note": " ".join(note),
            "images": [
                {"src": f"{p}_page.png", "caption": "CSP が保存した頁の見た目(赤い枠が、この層の在る場所)"},
                {"src": f"{p}_page_full.png", "caption": "同じ頁を原寸で重ねたもの"},
                {"src": f"{p}_layer.png", "caption": "この層だけ(白い背景に置いたもの)", "target": True},
            ],
        })

    spec = {
        "interview_id": "d3s_20261119" + ("_1axis" if a.one_axis else ""),
        "page_title": "名前が『線画』の層 33枚 —— 線が 3D から出たものかの裁定",
        "question": cfg["axis1"]["question"],
        "unit": "層",
        "decides": ("この層の線が 3D から出たものかが決まります"
                    "(名前では引けないので、1層ずつ決める約束になりました)"),
        "initial_view": "fit",
        "choices": ax1,
        "items": items,
    }
    if not a.one_axis:
        spec["question2"] = cfg["axis2"]["question"]
        spec["choices2"] = ax2
    out = IMGDIR / f"questions_{spec['interview_id']}.json"
    sys.path.insert(0, str(HERE / "tools/ui"))
    from interview_spec import load_questions
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    load_questions(out)

    print(f"問い {len(items)} / 画 {len(items) * 3}")
    print("軸1(線の出どころ)", [c["label"] for c in ax1], "+ 画面が足す(どれでもない・分からない)")
    print("軸2(何を描いたか)", "出さない(1列)" if a.one_axis else [c["label"] for c in ax2])
    print("初期値", "置いていない(機械の推測も、前の便の答えも無い)")
    print("札(原寸でほとんど見えない)", len(faint), faint)
    print("札(クリッピング)", len(clips), clips, "/ 見開き", len(spreads), spreads)
    print("頁", len(set(r["page_path"] for r in chk)))
    print("**a はこの便の答えを読まない** —— 名前で引けない 3D は a の外。a は 215,247 / 437,425 = 0.4921 のまま")
    print("出力", out)


main()
