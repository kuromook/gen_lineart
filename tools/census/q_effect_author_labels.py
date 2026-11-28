#!/usr/bin/env python
"""**名前 effect の層の作者の札の列(Q31〜Q33)**—— Track Q の頼み `note_q_census_effect_layer_author_labels_20261128.md`。
ユーザー 11-28: Q の案「作者の札を census の列に載せる(写しから引く 15層は『写しから』と書き分ける)」に「1」。

**作者の答えで、推測の列ではない。** 新しい走行はしない。札の語は聞き取りのボタンのまま、一言は逐語(規律6)。

- 作者の答え 68行: Q の表 `q_effect_layer_author_labels_q31_q33_20261128.csv`(sha256 先頭 6d38f6722ecd8642)。`label_source` = `作者`
- 写しの遅い方 15層: Q の表 `q_effect_ks1_copy_late_layers_20261128.csv`(sha256 先頭 9da7980540d0aa39)。
  写しの組は census の `page_file_copies_20261123.csv`(ファイルの sha の一致)。組の最も早い頁(`earliest_in_group`=1)の
  **同じ `layer_id`** を引き、構造 DB で層名が同じことを確かめ、その層に作者の札があれば写す。`label_source` = `写しから(作者は見ていない)`
  —— 写さなかった層は理由別に数えて印字する(規律9)
- **但し書き(列の説明)**: 20行の群は狙って引いた行。札の数を名前 effect の層の割合として引用しない。
  KS の当たる層は5つの作品フォルダーの中だけで、その外の黒消しは確かめていない

出力 `data/census_full/q31/q_effect_author_labels_20261128.csv`(非公開)。
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full", "q31")
ANS = os.path.join(D, "q_effect_layer_author_labels_q31_q33_20261128.csv")
LATE = os.path.join(D, "q_effect_ks1_copy_late_layers_20261128.csv")
COPIES = os.path.join(ROOT, "data", "census_full", "page_file_copies_20261123.csv")
OUT = os.path.join(D, "q_effect_author_labels_20261128.csv")
SHA = {ANS: "6d38f6722ecd8642", LATE: "9da7980540d0aa39"}


def main() -> int:
    print("[名前 effect の層の作者の札の列(Q31〜Q33、Track Q の頼み)]")
    for p, h in SHA.items():
        s = hashlib.sha256(open(p, "rb").read()).hexdigest()
        print(f"    {os.path.basename(p)} sha256 {s[:16]}(Q の値と一致: {s.startswith(h)})")
        if not s.startswith(h):
            return 1
    ans = list(csv.DictReader(open(ANS, encoding="utf-8")))
    lab = {(r["page_path"], int(r["layer_id"])): r for r in ans}
    assert len(lab) == len(ans), "作者の答えの鍵に重複"
    print(f"    作者の答え {len(ans)} 行: " + " / ".join(f"{k} {v}" for k, v in collections.Counter(r["author_label"] for r in ans).most_common()))
    cp = list(csv.DictReader(open(COPIES, encoding="utf-8")))
    grp_of = {r["page_path"]: r["copy_group"] for r in cp if r["copy_group"]}
    earliest = collections.defaultdict(list)
    for r in cp:
        if r["copy_group"] and r["earliest_in_group"] == "1":
            earliest[r["copy_group"]].append(r["page_path"])
    con = db.connect()
    name = {(p, i): (n or "") for p, i, n in con.execute("select page_path, layer_id, name from layers").fetchall()}

    out = [dict(page_path=r["page_path"], layer_id=int(r["layer_id"]), author_label=r["author_label"], comment=r["comment"],
                label_source="作者", registration=r["registration"], item_id=r["item_id"], copied_from_page="")
           for r in ans]
    drop = collections.Counter()
    late = list(csv.DictReader(open(LATE, encoding="utf-8")))
    for r in late:
        p, i = r["page_path"], int(r["layer_id"])
        g = grp_of.get(p)
        if g is None:
            drop["写しの組に入っていない"] += 1; continue
        es = [e for e in earliest[g] if e != p]
        if not es:
            drop["組の最も早い頁が自分だけ(遅い方ではない)"] += 1; continue
        hits = [e for e in es if (e, i) in lab]
        if not hits:
            drop["組の最も早い頁の同じ層に作者の札が無い"] += 1; continue
        if len({lab[(e, i)]["author_label"] for e in hits}) > 1:
            drop["最も早い頁が複数で札が食い違う"] += 1; continue
        e = sorted(hits)[0]
        if name.get((e, i)) != name.get((p, i)) or (p, i) not in name:
            drop["同じ layer_id の層名が違う・層が無い"] += 1; continue
        if (p, i) in lab:
            drop["遅い方の層にも作者の答えがある"] += 1; continue
        out.append(dict(page_path=p, layer_id=i, author_label=lab[(e, i)]["author_label"], comment="",
                        label_source="写しから(作者は見ていない)", registration=lab[(e, i)]["registration"],
                        item_id=lab[(e, i)]["item_id"], copied_from_page=e))
    n_cp = len(out) - len(ans)
    print(f"    写しの遅い方 {len(late)} 層 → 写した {n_cp} / 写さなかった {sum(drop.values())}")
    for k, v in drop.most_common():
        print(f"      {v:3d}  {k}")
    print("    写した層の札: " + " / ".join(f"{k} {v}" for k, v in collections.Counter(
        r["author_label"] for r in out if r["label_source"] != "作者").most_common()))
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"    → {OUT}({len(out)} 行、非公開)")
    print("    但し書き: 狙って引いた群の札。割合として引用しない。KS の当たる層は5つの作品フォルダーの中だけ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
