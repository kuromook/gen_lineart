# -*- coding: utf-8 -*-
"""**空(定義C)の条項を a の篩に足したら、何層増えるか** —— 前後を名前を付けて印字する。

全頁の画素が届く前に書いた器具である(**規律10**: 棒を動かす前に、動かし方を登録して印字する)。
届いたら同じ器具を全層の `empty_c` で回す —— 第2引数にその CSV(`page_path,layer_id,empty_c`)を渡す。

**何に反応しているかを名乗る**(条例9):
  入力は census の層ごとの判定 `a_verdict`(a の確定の表を当てた結果 + 測った `empty_c`)。
  `empty_c` が空欄の層は **測っていない**(母数0)。空でないとは言わない。
  名前と作者のラベルは役割の表 v2 から引く(**`text` 種別の名前は出さない。条例11**)。

**作者の裁定(2026-11-13、逐語)**: 「下絵　線画hair  blackも名前だけでは落とせない非決定論側だ
  これらが落とせるのは空の場合だけだ」 —— **名前で『除外しない』と答えた行も、空のときだけは落ちる。**
  段のラベルが付いた層も保留しない(保留した場合の数は参考として印字する)。

**全頁の画素の便から来る2つの保留**(census 2026-11-13、便1〜便7):
  - **`has_image_material` = 1 の層は空と読まない** —— 描画の塊が空でも元の画像を持つ(抽出側の但し書き)
  - **`pixel_formats` に形式5 を含む層は「調査中」**(2026-10-31 から。アルファが全面 255 と読まれる)——
    **調査中の層は分析対象外。外した単位は層×計測で、外した数を印字する**(作者の決定 2026-10-29)
  どちらも**空=1 でも落とさず、理由別に数を印字する**。

**印字するもの**:
  (1) 母集団と、空が測ってある範囲(**範囲の外は数に入れない**)
  (2) **前**(名前・種別・列だけ)と **後**(空の条項を足した)—— 両方に名前を付ける
  (3) 増えた層の理由別の内訳。**名前が『除外しない』の行の空の層**は別に数え、名前別に出す
  (4) **規律13 の点検**: 増えた層のうち、作者が段のラベルを付けた層が何層あるか
  (5) **画素が届いて動きうる層の数**(まだ測っていない `.clip` の、落ちていない層)
"""
import csv, collections, hashlib, io, sys

V = "inbox/a_verdict_20261112.csv"
ROLE = "data/role_predictor/layer_role_table_20261110_v2.csv"
STAGES = ("あたり", "下絵", "清書", "清書(線画)")


def sha(p):
    h = hashlib.sha256()
    with io.open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main(extra=None):
    print("判定: %s sha256 %s" % (V, sha(V)))
    if extra:
        print("空の列(全頁): %s sha256 %s" % (extra, sha(extra)))
    print("名前とラベル: %s sha256 %s\n" % (ROLE, sha(ROLE)))

    emp, hold = {}, {}
    if extra:
        for r in csv.DictReader(io.open(extra, encoding="utf-8")):
            k = (r["page_path"], r["layer_id"])
            if r.get("empty_c") in ("0", "1"):
                emp[k] = r["empty_c"]
            why = []
            if r.get("has_image_material") in ("1", "True", "true"):
                why.append("画像素材の層(空と読まない)")
            if "5" in [x.strip() for x in (r.get("pixel_formats") or "").split(";")]:
                why.append("形式5 を含む(調査中。分析対象外)")
            if why:
                hold[k] = " / ".join(why)

    rows = []
    for r in csv.DictReader(io.open(V, encoding="utf-8")):
        k = (r["page_path"], r["layer_id"])
        e = emp.get(k, r["empty_c"])
        rows.append((k, r["a_verdict"], e, r["page_path"].rsplit(".", 1)[-1].lower()))
    n_all = len(rows)

    meas = [x for x in rows if x[2] in ("0", "1")]
    e1 = [x for x in meas if x[2] == "1"]
    print("== (1) 母集団と、空が測ってある範囲 ==")
    print("   全 **%d 層**(構造 DB から恒久除外の頁の層を外した数)" % n_all)
    print("   空が測ってある層 **%d**(空=1 %d / 空=0 %d)/ **測っていない %d(母数0。空でないとは言わない)**"
          % (len(meas), len(e1), len(meas) - len(e1), n_all - len(meas)))
    ext = collections.Counter(x[3] for x in meas)
    print("   測ってある層の拡張子: " + " / ".join("%s %d" % (k, v) for k, v in ext.most_common()))
    if not extra:
        print("   **範囲は層の役割の表 v3 の層だけ** —— 名前で引いた母集団(raster と vector、塗りはほぼ入らない)。")
        print("   **だから空の率 %d/%d = %.4f を全層に引き伸ばしてはいけない**" % (len(e1), len(meas), len(e1) / len(meas)))

    nm = {}
    for r in csv.DictReader(io.open(ROLE, encoding="utf-8")):
        nm[(r["page_path"], r["layer_id"])] = r

    def stage(k):
        return nm[k]["author_stage"] if k in nm else ""

    held_rows = [x for x in rows if x[2] == "1" and x[1] != "template_excluded" and x[0] in hold]
    if extra:
        print("== (1b) 空=1 でも落とさない層(**理由別に数える**。census の列) ==")
        if held_rows:
            cnt = collections.Counter(hold[x[0]] for x in held_rows)
            for k, v in cnt.most_common():
                print("   %-40s **%d 層**" % (k, v))
            print("   ---- 計 **%d 層**を空から外す" % len(held_rows))
        else:
            print("   0 層(この入力には `has_image_material` / `pixel_formats` の印が無い)")
        print()

    before = sum(1 for x in rows if x[1] == "template_excluded")
    e1 = [x for x in e1 if x[0] not in hold]
    add_und = sum(1 for x in e1 if x[1] == "undecided")
    add_not = sum(1 for x in e1 if x[1] == "not_excluded")
    add_already = sum(1 for x in e1 if x[1] == "template_excluded")
    after = before + add_und + add_not
    print("\n== (2) 前と後(**両方に名前を付ける**) ==")
    print("   **前: 名前と種別と列だけ** = %d / %d = **%.4f**" % (before, n_all, before / n_all))
    print("   **後: 空(定義C)の条項を足した** = %d / %d = **%.4f**" % (after, n_all, after / n_all))
    print("   差 **+%d(+%.4f)** —— **測ってある範囲でだけの差である**" % (after - before, (after - before) / n_all))
    held = sum(1 for k, v, e, _ in e1 if v != "template_excluded" and stage(k))
    after_h = after - held
    print("   (参考)作者の段のラベルが付いた %d 層を保留した場合 = %d / %d = %.4f"
          % (held, after_h, n_all, after_h / n_all))
    print("   **当てる規則は『後』の方である** —— 2026-11-13 の作者の裁定(逐語)")
    print("   **「下絵　線画hair  blackも名前だけでは落とせない非決定論側だ　これらが落とせるのは空の場合だけだ」**")
    print("   **名前の答えも段のラベルも、空の層を守らない。** 保留は作者の裁定で解けた")
    print("   (**作者の裁定は点数と独立だから、点検を緩めたことにはならない** —— 規律9-5)")

    print("\n== (3) 増えた層の内訳(規律9: 理由別) ==")
    print("   空=1 で、まだ決まっていなかった層        **%d**" % add_und)
    print("   空=1 で、名前の答えが『除外しない』の層  **%d**(E の決め。作者の言葉ではない)" % add_not)
    print("   空=1 で、既に落ちていた層                %d(重なり)" % add_already)

    c = collections.Counter(); miss = 0
    for k, verdict, e, _ in e1:
        if verdict != "not_excluded":
            continue
        if k in nm:
            a, b = nm[k]["name"], nm[k]["v_layer_kind"]
            c[("(名前は出さない。条例11)" if b == "text" else (a or "(空の名前)"), b)] += 1
        else:
            miss += 1
    print("   その名前(引けなかった %d):" % miss)
    for (a, b), v in c.most_common():
        print("      %-18s %-8s %5d" % (a, b, v))

    print("\n== (4) 規律13 の点検: 増えた層に作者の段のラベルが付いているか ==")
    lab = collections.Counter(); nolab = 0
    for k, verdict, e, _ in e1:
        if verdict == "template_excluded":
            continue
        s = stage(k)
        if s:
            lab[s] += 1
        else:
            nolab += 1
    if not lab:
        print("   ラベルの付いた層は **0層**。**ラベルの無い %d 層については何も示していない**" % nolab)
    else:
        print("   **ラベルの付いた層が %d 層ある** —— 内訳:" % sum(lab.values()))
        for k, v in lab.most_common():
            print("      %-28s %4d" % (k, v))
        bad = sum(v for k, v in lab.items() if any(s in k for s in STAGES))
        print("   **うち『あたり・下絵・清書』の層: %d**" % bad)
        if bad:
            print("   → **空の条項は、作者が段を付けた層を落とす。** 数を隠さず、作者が覆せる形で出す")
        print("   (ラベルの無い %d 層については何も示していない)" % nolab)
        # **作者自身のコメント** —— 段のラベルと空の印が衝突しているのかを、作者の言葉で見る
        agree = collections.Counter(); other = collections.Counter()
        for k, verdict, e, _ in e1:
            if verdict == "template_excluded":
                continue
            r = nm.get(k)
            if not r or not r["author_stage"]:
                continue
            cm = (r["author_comment"] or "").strip()
            key = (r["author_stage"], r["name"] if r["v_layer_kind"] != "text" else "(text)", cm)
            (agree if ("空" in cm or "何もない" in cm) else other)[key] += 1
        print("\n   **作者自身のコメント**(規律6: 作者の答えは書き換えない。並べて読む)")
        print("   | 段のラベル | 名前 | 作者のコメント | 層 |")
        print("   |---|---|---|---:|")
        for d in (agree, other):
            for (st, nmx, cm), v in d.most_common():
                print("   | %s | `%s` | %s | %d |" % (st, nmx, cm if cm else "(コメントなし)", v))
        na = sum(agree.values()); no = sum(other.values())
        print("   **作者が自分で『空』『何もないレイヤ』と書いた層: %d / コメントが別の層: %d**" % (na, no))
        print("   → **段のラベルは層の名前と役目に付いた答えで、空の印は中身に付いた測定である。**")
        print("   **%d 層では、作者の2つの答えが測定と揃っている**(段を答え、かつ空だと書いた)" % na)

    print("\n== (5) 画素が届いて動きうる層 ==")
    movable = sum(1 for k, verdict, e, x in rows
                  if verdict == "undecided" and e not in ("0", "1") and x == "clip")
    lip_und = sum(1 for k, verdict, e, x in rows if verdict == "undecided" and x == "lip")
    print("   まだ落ちていない `.clip` の層で、空を測っていない層 **%d**(ここが動きうる)" % movable)
    print("   まだ落ちていない `.lip` の層 **%d** —— **画素が届いても測れない**(母数0)" % lip_und)

    print("\n== (6) 条項を当てた後の a の状態(全層) ==")
    e1set = set(x[0] for x in e1)
    st = collections.Counter(); out = []
    for k, v, e, x in rows:
        if v == "template_excluded":
            s2, r2 = "template_excluded", "名前・種別・列"
        elif k in e1set:
            s2, r2 = "template_excluded", "条項:空(定義C)"
        elif k in hold and e == "1":
            s2, r2 = ("not_excluded" if v == "not_excluded" else "undecided"), "空だが保留(%s)" % hold[k]
        else:
            s2, r2 = v, ""
        st[(s2, r2)] += 1
        out.append((k[0], k[1], s2, r2))
    tot_x = sum(v for (a, _), v in st.items() if a == "template_excluded")
    print("   | 状態 | 理由 | 層 |")
    print("   |---|---|---:|")
    for (a, r), v in sorted(st.items(), key=lambda z: -z[1]):
        print("   | %s | %s | %d |" % (a, r if r else "—", v))
    print("   **落とす 計 %d / %d = %.4f**" % (tot_x, n_all, tot_x / n_all))
    if len(sys.argv) > 2:
        p = sys.argv[2]
        with io.open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["page_path", "layer_id", "a_verdict_v2", "a_rule_v2"])
            w.writerows(out)
        print("   層ごとの判定を書いた: %s(%d 行。**`data/` の外に出さない**)" % (p, len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
