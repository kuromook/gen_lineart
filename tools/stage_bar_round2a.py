# -*- coding: utf-8 -*-
"""**便A の段の棒を読む**(5束97行がそろってから読むと登録してあった。登録簿
`role_predictor.a_bundles.template_choice_added.populations_to_print` / `two_bars`)。

**下限の棒 = census の名前の当て方**(`name_rule`。作者のものではない)。**本当の棒は作者の判定**。
**帰無値は2つ印字する** —— 多数派の段と名前の当て方。片方だけでは勝ちにならない。

**印字する母集団は4つ**(走る前に決めた): 97行 / **92行(空を外す。主)** /
旧い選択肢の束だけ / 新しい選択肢の束だけ。**3と4を並べないと選択肢の違いが点数の差に化ける**。

**『テンプレート』は段の棒では「段ではない」に写す**(読む前に決めた)。
層名は印字しない(条例11)。使い方: ./venv/bin/python tools/stage_bar_round2a.py
"""
import csv, collections, hashlib, io, os, sys

# Track R: E の 91eed9e から写した。変えたのは入力の在りか(環境変数で役割の表の版を選ぶ)だけ
ANS = os.environ.get("STAGE_BAR_ANS", "data/census/hearing_round2_a_answers.csv")
ROLE = os.environ.get("STAGE_BAR_ROLE", "data/census/layer_role_table_20261112_v3.csv")
# 作者の選択肢 → 段の語彙。**テンプレートは「段ではない」に写す**(登録済み)
STAGE = {"あたり": "あたり", "下絵": "下絵", "清書(線画)": "清書",
         "段の線ではない(塗り・効果・資料など)": "段ではない",
         "テンプレート(トンボ・裁ち落とし枠など、印刷のための層)": "段ではない",
         "分からない": "分からない"}
PRED = {"あたり": "あたり", "下絵": "下絵", "清書": "清書", "なし": "なし"}
ORDER = ("あたり", "下絵", "清書", "段ではない")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def read():
    a = list(csv.DictReader(io.open(ANS, encoding="utf-8")))
    rt = {(r["page_path"], r["layer_id"]): r for r in csv.DictReader(io.open(ROLE, encoding="utf-8"))}
    out, miss = [], 0
    for r in a:
        k = (r["page_path"], r["layer_id"])
        if k not in rt:
            miss += 1
            continue
        t = rt[k]
        raw = r["author_stage"]
        out.append({"bundle": r["bundle"], "pos": int(r["pos"]), "empty": r["empty"] == "1",
                    "raw": raw, "stage": STAGE.get(raw, "**写せない: %s**" % raw),
                    "pred": PRED.get(t["name_rule"], "**知らない予測: %s**" % t["name_rule"]),
                    "kind": t["v_layer_kind"], "unit": t["split_unit"],
                    "comment": r["author_comment"]})
    return out, miss


def bar(rows, label):
    print("\n" + "=" * 70)
    print("== 母集団: **%s** —— %d 行 ==" % (label, len(rows)))
    if not rows:
        print("   **母数0。測れない**(登録の文のまま: 母数0 = unmeasurable)")
        return
    st = collections.Counter(r["stage"] for r in rows)
    print("   作者の答え: %s" % " / ".join("%s %d" % (k, st[k]) for k in list(ORDER) + ["分からない"] if st.get(k)))
    unk = [r for r in rows if r["stage"] == "分からない"]
    use = [r for r in rows if r["stage"] in ORDER]
    print("   **落とした行(規律9)**: 分からない %d 行 —— 棒からは外す。**答えは書き換えない**" % len(unk))
    if unk:
        print("     その内訳: 空 %d / 空でない %d / コメント: %s"
              % (sum(1 for r in unk if r["empty"]), sum(1 for r in unk if not r["empty"]),
                 dict(collections.Counter((r["comment"] or "—")[:12] for r in unk))))
    print("   ---- 棒を読む行: **%d**" % len(use))
    if not use:
        return

    # 帰無値1: 多数派の段
    maj, majn = collections.Counter(r["stage"] for r in use).most_common(1)[0]
    print("\n   **帰無値1(多数派の段)**: 「%s」と全部答える → %d / %d = **%.4f**" % (maj, majn, len(use), majn / len(use)))

    # 帰無値2 = 下限の棒: 名前の当て方
    pred_on = [r for r in use if r["pred"] != "なし"]
    hit = sum(1 for r in pred_on if r["pred"] == r["stage"])
    print("   **帰無値2 = 下限の棒(census の名前の当て方)**")
    print("     名前の当て方が**何か言う行**: %d / %d。そのうち当たり **%d = %.4f**"
          % (len(pred_on), len(use), hit, hit / len(pred_on) if pred_on else float("nan")))
    print("     名前の当て方が**「なし」の行**: %d —— **予測をしていない(判定保留)**。その行の作者の答え: %s"
          % (len(use) - len(pred_on), dict(collections.Counter(r["stage"] for r in use if r["pred"] == "なし"))))
    print("     **全行を母数にすると** 当たり %d / %d = **%.4f**(「なし」を外れと数えた場合の下界)"
          % (hit, len(use), hit / len(use)))

    print("\n   段ごと(**的中率と取りこぼしを両方印字する**。規律13)")
    print("   | 段 | 作者の答え | 名前が当てた | 的中率 | 取りこぼし |")
    print("   |---|---:|---:|---:|---:|")
    for s in ORDER:
        n_a = sum(1 for r in use if r["stage"] == s)
        n_p = sum(1 for r in use if r["pred"] == s)
        tp = sum(1 for r in use if r["pred"] == s and r["stage"] == s)
        prec = "%.4f" % (tp / n_p) if n_p else "**母数0**"
        miss = "%.4f" % ((n_a - tp) / n_a) if n_a else "**母数0**"
        print("   | %s | %d | %d | %s | %s |" % (s, n_a, n_p, prec, miss))
    print("   **「段ではない」は名前の当て方に予測が無い**(語彙が 清書/下絵/なし だけ)。だから母数0 になる")

    print("\n   混同(行=作者、列=名前の当て方)")
    cols = ["あたり", "下絵", "清書", "なし"]
    print("   | 作者＼名前 | %s |" % " | ".join(cols))
    print("   |---|%s" % ("---:|" * len(cols)))
    for s in ORDER:
        print("   | %s | %s |" % (s, " | ".join(
            str(sum(1 for r in use if r["stage"] == s and r["pred"] == c)) for c in cols)))


def main():
    print("答え  : %s sha256 %s" % (ANS, sha(ANS)))
    print("役割表: %s sha256 %s" % (ROLE, sha(ROLE)))
    rows, miss = read()
    print("\n便A: %d 行(役割の表に無かった行: %d)" % (len(rows), miss))
    bad = [r for r in rows if r["stage"].startswith("**") or r["pred"].startswith("**")]
    if bad:
        print("**写せない値が在る。止める**: %s" % collections.Counter(r["raw"] for r in bad))
        return 1

    print("\n== **選択肢『テンプレート』が作者に届いたか** ==")
    tmpl = [r for r in rows if r["raw"].startswith("テンプレート")]
    print("   選択肢『テンプレート』を選んだ行: **%d**" % len(tmpl))
    cm = [r for r in rows if "テンプレート" in (r["comment"] or "")]
    print("   コメントに『テンプレート』と書いた行: **%d** —— %s"
          % (len(cm), [(r["bundle"], r["pos"], r["stage"]) for r in cm]))
    print("   → **97行の答えの語彙に『テンプレート』が1つも無い。** 選択肢は便A の作者には**届いていない**")
    print("     (登録の『新しい選択肢の束だけ』は **母数0 = 測れない**。A-5 の2行は選択肢が無く『分からない』+コメント)")
    print("   **これは選択肢の欠けの3件目**(1件目 A-1 の『空』、2件目 A-2 のテンプレート)。**作者は物を分かっていて、入れる箱が無かった**")

    bar(rows, "97行(空を入れる)")
    bar([r for r in rows if not r["empty"]], "92行(空を外す。**主**)")
    bar(rows, "旧い選択肢の束だけ(= 97行。便A の全部が旧い選択肢だった)")
    bar([r for r in rows if r["raw"].startswith("テンプレート")], "新しい選択肢の束だけ")

    print("\n" + "=" * 70)
    print("== 名前の当て方の誤りの実例(登録済み。**数は棒と一緒に読む**) ==")
    for r in rows:
        if r["pred"] == "清書" and r["stage"] == "段ではない":
            print("   束=%s pos=%-2d 名前の当て方=清書 / 作者=段ではない / コメント=%s" % (r["bundle"], r["pos"], r["comment"] or "—"))
    print("   → **名前が `線画` でも中身がベタの層が在る**(A-2 pos25、コメント「髪ベタ」)。**名前の当て方の誤りの実例**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
