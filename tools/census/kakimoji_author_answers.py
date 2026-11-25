#!/usr/bin/env python
"""**作者が層について「描き文字」と答えた行の一覧**(Track Q の頼み `note_q_census_kakimoji_author_answers_inventory_20261125.md`、
2026-11-25。**新しい走行ではない** —— 既に在る作者の答えを集めて数えるだけ)。

在りか(層の鍵 = 頁・層 id を持つ答え。どれも作者の答えで、機械の初期値の行は落として数える):
- `stage1`  段の聞き取り 第1便(役割の表 v3 の `author_stage` / `author_comment`、聞き取りの 265行)
- `round2A` 基盤の聞き取り 第2巡 便A(`hearing_round2_a_answers.csv`)
- `round2B` 同 便B(`hearing_round2_b_answers.csv`、`answer_source = author` の行だけ)
- `suspect33` succor の 3D の疑いの 33層(`succor_3d_suspect33_answers_for_census_20261119.csv`)
- `white` 白い層の聞き取り(`white_hearing/answers_latest_20261123.csv`、項目 wh_NN = 一覧の pos NN)
- `q29` Q29 の作者の20行(`q29/q29_author_layer_answers_20261125.csv`)
- 探して当たりの無かった在りか(資料の85層・image_material の50+50・線画の塗りの聞き取り)は 0 行と印字する

答えの読み方(**答えを言い換えない。一致させるのは語だけ**、規律6):
- `描き文字と答えた` = 答えの語(選んだ札・一言)に 描き文字・書き文字・かき文字 が在る(「描き文字のあたり」も入る。逐語で区別できる)
- `フキダシか描き文字(1つのボタン)` = 白い層の聞き取りの札「フキダシ・描き文字なので別の区分」。**どちらかは分からない**
- `描き文字の語でない文字` = 一言に「文字」が在るが上の語でない(例: 看板の文字)
- `名前は描き文字・答えは別` = 層の名前か祖先のフォルダー名に描き文字の語が在り、答えに上のどれも無い(名前の当て方の誤検出の材料)

名前の群: 描き文字の名前(層名か祖先)/ effect(層名に effect、大小を問わない)/ 線画(層名がちょうど 線画)/ その他。
**線画に直接描き込まれた描き文字か**(`in_lineart`)は答えの一言にそう書かれているときだけ 1、他は空(分からない)。
同じ層が複数の在りかに在れば1行に畳み、在りかを `;` で並べる(畳んだ数を印字する)。テキストの層(`v_layer_kind = text`)は別に数える。
**名前で引ける 描き文字・フキダシ 887層は正解の数ではない。この一覧と並べるときは「名前の下限の材料」と書く。**

出力 `data/census_full/kakimoji_author_answers_20261125.csv`(非公開)。
"""
import collections
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
OUT = os.path.join(D, "kakimoji_author_answers_20261125.csv")
KAKI = re.compile(r"(描き文字|書き文字|かき文字|描文字)")
MOJI = re.compile(r"文字")
EMPTY = [("im85", "shiryou_folder_sketch_im85_two_axis_answers_20261118.csv"),
         ("image_material50", "image_material_check50_answers_v2_last_20261115.csv"),
         ("image_material50b", "image_material_check50b_answers_20261116.jsonl"),
         ("lineart_fill", "lineart_fill_answers_20261107.csv")]


def rd(name):
    return list(csv.DictReader(open(os.path.join(D, name), encoding="utf-8")))


def harvest(drop):
    """(在りか, 行の印, page_path, layer_id, 答えの逐語) を返す。"""
    out = []
    for r in rd("layer_role_table_20261112_v3.csv"):
        if r["author_stage"] or r["author_comment"]:
            out.append(("stage1", "役割の表 v3", r["page_path"], r["layer_id"], f"{r['author_stage']}|{r['author_comment']}"))
    for r in rd("hearing_round2_a_answers.csv"):
        out.append(("round2A", f"{r['bundle']} 項{r['item']} 位{r['pos']}", r["page_path"], r["layer_id"], f"{r['author_stage']}|{r['author_comment']}"))
    for r in rd("hearing_round2_b_answers.csv"):
        if r["answer_source"] != "author":
            drop[f"round2B: 作者の答えでない行({r['answer_source']})"] += 1; continue
        out.append(("round2B", f"{r['bundle']} 項{r['item']} 位{r['pos']}", r["page_path"], r["layer_id"], f"{r['stage']}|{r['author_comment']}"))
    for r in rd("succor_3d_suspect33_answers_for_census_20261119.csv"):
        out.append(("suspect33", f"位{r['pos']}", r["page_path"], r["layer_id"], f"{r['label_origin']}|{r['label_draws']}|{r['comment']}"))
    lab = {c["value"]: c["label"] for c in json.load(open(os.path.join(D, "white_hearing", "questions.json"), encoding="utf-8"))["choices"]}
    wl = {int(r["pos"]): r for r in csv.DictReader(open(os.path.join(ROOT, "outbox", "white_exception_hearing_list_20261122.csv"), encoding="utf-8"))}
    for r in rd("white_hearing/answers_latest_20261123.csv"):
        k = wl[int(r["item_id"].split("_")[1])]
        out.append(("white", r["item_id"], k["page_path"], k["layer_id"], f"{lab.get(r['value'], r['value'])}|{r['comment']}"))
    for r in rd("q29/q29_author_layer_answers_20261125.csv"):
        out.append(("q29", r["item_id"], r["page_path"], r["layer_id"], f"{r['answer_label']}|{r['author_comment']}"))
    return out


def main() -> int:
    drop = collections.Counter()
    rows = harvest(drop)
    for tag, f in EMPTY:
        txt = open(os.path.join(D, f), encoding="utf-8").read()
        print(f"在りか {tag}: 文字の語 {len(MOJI.findall(txt))}(当たり無し、行を作らない)")
    print("答えの行: " + " / ".join(f"{k} {v}" for k, v in collections.Counter(r[0] for r in rows).items()))
    c = db.connect()
    keys = sorted({(r[2], int(r[3])) for r in rows})
    info = {}
    for p, i in keys:
        x = c.execute("select coalesce(l.name,''), v.v_layer_kind, l.parent_id, pg.year from layers l join v_layer_kind v using (page_path, layer_id) "
                      "join pages pg using (page_path) where l.page_path = ? and l.layer_id = ?", [p, i]).fetchone()
        if x is None:
            continue
        anc, q = [], x[2]
        while q is not None:
            a = c.execute("select coalesce(name,''), parent_id from layers where page_path = ? and layer_id = ?", [p, q]).fetchone()
            if a is None:
                break
            anc.append(a[0]); q = a[1]
        info[(p, i)] = dict(name=x[0], kind=x[1], year=x[3], anc="/".join(reversed(anc)))
    fold = collections.defaultdict(list)
    for src, where, p, i, ans in rows:
        k = (p, int(i))
        if k not in info:
            drop[f"{src}: DB に層が無い"] += 1; continue
        nm = info[k]["name"] + "/" + info[k]["anc"]
        if src == "white" and ans.startswith("フキダシ・描き文字"):
            m = "フキダシか描き文字(1つのボタン)"
        elif KAKI.search(ans):
            m = "描き文字と答えた"
        elif MOJI.search(ans):
            m = "描き文字の語でない文字"
        elif KAKI.search(nm):
            m = "名前は描き文字・答えは別"
        else:
            continue
        fold[k].append((src, where, ans, m))
    order = ["描き文字と答えた", "フキダシか描き文字(1つのボタン)", "描き文字の語でない文字", "名前は描き文字・答えは別"]
    out = []
    for k, hs in sorted(fold.items()):
        f = info[k]
        nm = f["name"]
        grp = ("描き文字の名前" if KAKI.search(nm + "/" + f["anc"]) else "effect" if "effect" in nm.lower()
               else "線画" if nm == "線画" else "その他")
        ms = sorted({h[3] for h in hs}, key=order.index)
        out.append(dict(page_path=k[0], layer_id=k[1], layer_name=nm, ancestors=f["anc"], v_layer_kind=f["kind"], year_label=f["year"],
                        name_group=grp, match=ms[0], match_all=";".join(ms),
                        author_answer=" ;; ".join(h[2] for h in hs), sources=";".join(f"{h[0]}:{h[1]}" for h in hs),
                        n_sources=len(hs),
                        in_lineart=int(any(re.search(r"線画に(直接)?(描|書)", h[2]) for h in hs)) or ""))
    print("落とした: " + (" / ".join(f"{k} {v}" for k, v in drop.items()) or "0"))
    nh = sum(len(v) for v in fold.values())
    print(f"当たった答えの行 {nh} → 層 {len(out)}(畳んだ {nh - len(out)}。複数の在りかに在る層 {sum(o['n_sources'] > 1 for o in out)})")
    for m in order:
        s = [o for o in out if o["match"] == m]
        print(f"\n[{m}] 層 {len(s)}(うちテキストの層 {sum(o['v_layer_kind'] == 'text' for o in s)})")
        if not s:
            continue
        src = collections.Counter(x.split(":")[0] for o in s for x in o["sources"].split(";"))
        print("  在りか(畳む前): " + " / ".join(f"{k} {v}" for k, v in src.most_common()))
        print("  名前の群: " + " / ".join(f"{k} {v}" for k, v in collections.Counter(o["name_group"] for o in s).most_common()))
        print("  年代札: " + " / ".join(f"{k} {v}" for k, v in sorted(collections.Counter(o["year_label"] for o in s).items())))
        print("  種別: " + " / ".join(f"{k} {v}" for k, v in collections.Counter(o["v_layer_kind"] for o in s).most_common()))
    print(f"\n線画に直接描き込まれたと一言に在る層: {sum(o['in_lineart'] == 1 for o in out)}")
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"出力: {OUT}({len(out)}行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
