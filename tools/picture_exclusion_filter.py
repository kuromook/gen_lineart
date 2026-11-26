# -*- coding: utf-8 -*-
"""「絵から除外」する層の決定論的な篩(第2段)。

表 `config/picture_exclusion_draft_20261111.tsv` の **状態=確定 の行だけ**を効かせる。
**候補・材料・保留・測れない・裁定あり の行は効かせない**(表は「安全なものの表」ではなく
「選んだものの表」。足すのはユーザーの承認)。

規律9: **落としたものを理由別に数えて印字する。** さらに規律13 の点検として、
**落ちた層に作者のラベルが在るなら、その段を印字する** —— 段のラベルが在る層を落としたら、
それは反証であって篩の成功ではない。

層名は印字しない(条例11。`text` 種別の名前は本文で、奥付に作者の個人情報が入る)。
使い方: ./venv/bin/python tools/picture_exclusion_filter.py [表のCSV]
"""
import csv, collections, hashlib, io, os, sys

TSV = "config/picture_exclusion_draft_20261111.tsv"
DEFAULT_TABLE = "data/role_predictor/layer_role_table_20261110_v2.csv"
STAGES = ("あたり", "下絵", "清書", "段ではない")

# **規律13 の点検の例外は、作者の裁定だけが作れる。** 器具が自分で緩めてはいけない。
# 1件ごとに (頁, 層) と逐語を書き、**発火したかどうかに関わらず毎回印字する**
# —— 黙って緩む点検は、点検が無いより悪い(「名前は点検ではない」)。
AUTHOR_RULED_EXCEPTIONS = {
    ("F:\\manuscripts\\2016\\suc\\subbcubussBK\\page0008.clip", "186"): (
        "2026-11-23 / census の問いへの答え: **「この層にも当てる」**"
        "(選択肢の説明: 前の『清書』は誤り。この層もベースマスク)。"
        "**前の段のラベル『清書』は役割の表に残す。消さない・書き換えない**(規律10)"),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load_rules(path):
    """確定の行だけを返す。効かせない行も、理由を付けて返す(印字するため)。"""
    on, off = [], []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            c = line.split("\t")
            if c[0] == "決め方":
                continue
            r = {"how": c[0], "key": c[1], "kind": c[2], "state": c[-1]}
            (on if r["state"] == "確定" else off).append(r)
    return on, off


# 表の鍵(名前・種別)から、層の役割の表の列への当て方。
# **名前は完全一致**(棚卸しと同じ。全角半角や空白を畳まない)。
KIND_KEYS = {
    "用紙": {"paper"},
    "(ノンブル)・(作品情報)": {"story_info"},
    "セリフのテキスト層": {"text"},
}


def match(rule, row):
    if rule["how"] == "名前":
        return row["name"] == rule["key"] and row["v_layer_kind"] == rule["kind"]
    if rule["how"] == "種別":
        return row["v_layer_kind"] in KIND_KEYS.get(rule["key"], set())
    return False


def main(table=DEFAULT_TABLE):
    print("表(篩)   :", TSV, "sha256", sha(TSV))
    print("表(母集団):", table, "sha256", sha(table))
    on, off = load_rules(TSV)
    col = [r for r in on if r["how"] == "列"]
    cls = [r for r in on if r["how"] == "条項"]
    on = [r for r in on if r["how"] not in ("列", "条項")]
    print("\n効かせる行(状態=確定): %d" % len(on))
    for r in on:
        print("   %s / %s / %s" % (r["how"], r["key"], r["kind"]))
    if col:
        print("**器具が当てられない確定の行: %d**(決め方=列。棚卸しにも役割の表にもこの列が無い)" % len(col))
        for r in col:
            print("   列 / %s / %s —— **census の数で別に足す**" % (r["key"], r["kind"]))
    print("効かせない行: %d" % len(off))
    for r in off:
        print("   [%s] %s / %s" % (r["state"], r["how"], r["key"]))

    rows = list(csv.DictReader(io.open(table, encoding="utf-8")))
    print("\n母集団: %d 層" % len(rows))
    print("**この母集団は名前で引いた層と聞き取りの層で、全437,425層ではない。**")

    dropped = collections.Counter()
    dropped_labelled = collections.defaultdict(collections.Counter)
    ruled_exempt = collections.defaultdict(collections.Counter)
    multi = 0
    kept = []
    for row in rows:
        hits = [r for r in on if match(r, row)]
        if not hits:
            kept.append(row)
            continue
        if len(hits) > 1:
            multi += 1
        key = "%s: %s" % (hits[0]["how"], hits[0]["key"])
        dropped[key] += 1
        st = row.get("author_stage", "")
        if st:
            dropped_labelled[key][st] += 1
            if (row["page_path"], row["layer_id"]) in AUTHOR_RULED_EXCEPTIONS:
                ruled_exempt[key][st] += 1

    print("\n== 落とした層(理由別。規律9) ==")
    print("| 理由 | 落とした層 | うち作者のラベル在り | その段 |")
    print("|---|---:|---:|---|")
    tot = 0
    bad = []
    for key in sorted(dropped, key=lambda k: -dropped[k]):
        lab = dropped_labelled[key]
        n_lab = sum(lab.values())
        stages = " / ".join("%s %d" % (s, lab[s]) for s in STAGES if lab.get(s))
        print("| %s | %d | %d | %s |" % (key, dropped[key], n_lab, stages or "—"))
        tot += dropped[key]
        for s in ("あたり", "下絵", "清書"):
            if lab.get(s):
                n_ex = ruled_exempt[key].get(s, 0)
                if lab[s] - n_ex > 0:
                    bad.append((key, s, lab[s] - n_ex, n_ex))
    print("| 計 | %d | %d | |" % (tot, sum(sum(v.values()) for v in dropped_labelled.values())))
    if multi:
        print("2つ以上の理由に当たった層: %d(上の表では最初の理由に数えた)" % multi)

    print("\n== 残った層: %d(%d - %d) ==" % (len(kept), len(rows), tot))
    lab_kept = collections.Counter(r["author_stage"] for r in kept if r.get("author_stage"))
    print("   残った層の作者のラベル:", dict(lab_kept))

    print("\n== 規律13 の点検 ==")
    print("作者の裁定による例外: **%d 件**(器具は自分で例外を作らない)" % len(AUTHOR_RULED_EXCEPTIONS))
    for (pp, lid), words in sorted(AUTHOR_RULED_EXCEPTIONS.items()):
        print("   %s 層 %s —— %s" % (pp, lid, words))
    n_fired = sum(sum(v.values()) for v in ruled_exempt.values())
    print("   **この走行で例外に当たって落ちた層: %d**"
          "(例外が無ければ点検は発火していた)" % n_fired)
    if bad:
        print("**反証**: 作者が段と答えた層を落としている。篩を止めること:")
        for key, s, n, n_ex in bad:
            print("   %s —— %s を %d 層(裁定の例外 %d を除いた数)" % (key, s, n, n_ex))
        return 1
    print("落とした層に「あたり・下絵・清書」のラベルは、**裁定の例外 %d 層を除いて 0層**。"
          % n_fired)
    print("(ラベルが在るのは %d 層で、全部「段ではない」。**ラベルの無い層については何も示していない**)"
          % sum(sum(v.values()) for v in dropped_labelled.values()))
    return 0


INVENTORY = "data/role_predictor/layer_name_inventory_20261110.csv"


def corpus(path=INVENTORY):
    """全層の棚卸し(437,425層)に篩を当てて、作品全体で何層落ちるかを数える。
    **役割の表の母集団は名前で引いた層なので、全体の数はこちらで数える。**"""
    print("\n" + "=" * 68)
    print("== 作品全体(棚卸し)で何層落ちるか ==")
    print("棚卸し:", path, "sha256", sha(path))
    on, _ = load_rules(TSV)
    col = [r for r in on if r["how"] == "列"]
    cls = [r for r in on if r["how"] == "条項"]
    on = [r for r in on if r["how"] not in ("列", "条項")]
    rows = list(csv.DictReader(io.open(path, encoding="utf-8")))
    n_all = sum(int(r["n_layers"]) for r in rows)
    print("棚卸しの行: %d / 層: %d" % (len(rows), n_all))

    def m(rule, r):
        if rule["how"] == "名前":
            return r["layer_name"] == rule["key"] and r["v_layer_kind"] == rule["kind"]
        if rule["how"] == "種別":
            return r["v_layer_kind"] in KIND_KEYS.get(rule["key"], set())
        return False

    print("\n| 理由 | 棚卸しの行 | 落とす層 | 頁 | 作品単位 |")
    print("|---|---:|---:|---:|---:|")
    tot = 0
    seen = set()
    for rule in on:
        hit = [r for r in rows if m(rule, r)]
        n = sum(int(r["n_layers"]) for r in hit)
        pg = sum(int(r["n_pages"]) for r in hit)
        u = max([int(r["n_split_units"]) for r in hit] or [0])
        print("| %s: %s | %d | %d | %d | %d |" % (rule["how"], rule["key"], len(hit), n, pg, u))
        tot += n
        for r in hit:
            seen.add((r["layer_name"], r["v_layer_kind"]))
    print("| 計 | %d | **%d** | | |" % (len(seen), tot))
    print("\n落とす層 %d / 全 %d = %.4f **(決め方=列 の行を含まない)**" % (tot, n_all, tot / n_all))
    if col:
        CENSUS_COL = {"tone_enabled = 1": 91060}
        print("\n**決め方=列 の行(器具では当てられない。census の数)**")
        add = 0
        for r in col:
            n = CENSUS_COL.get(r["key"])
            print("   列: %s —— %s 層(census 2026-11-12)" % (r["key"], n if n is not None else "未測定"))
            add += n or 0
        print("   %d + %d = **%d / %d = %.4f**" % (tot, add, tot + add, n_all, (tot + add) / n_all))
        print("   **重なりは測った値として0である**(census、2026-11-12、全437,425層) —— `tone_enabled`=1 の 91,060層は")
        print("   **確定の名前・種別のどの行にも当たらず、`調整` との重なりも0**。2つ以上の行に当たる層も0。")
        print("   **『重なり0の仮定の下の値』という但し書きは外した。** census の層ごとの判定 `a_verdict_20261112.csv` と")
        print("   突き合わせた: template_excluded 196,291 / not_excluded 24,801 / undecided 216,333、")
        print("   うち列で落ちた 91,060 を引くと **105,231 = 当時(2026-11-12)の この器具の名前と種別の数と一致**")
        print("   (**いまの名前と種別の数は上の表の計** —— 11-23 のベースマスク 5,148 と 11-26 の `下地` 161 が足された。")
        print("   **当時の一致はその日の版についてのもので、いまの数との一致を主張しない**)")
    if cls:
        print("\n**決め方=条項 の行(層ごとの測定が要る。器具では当てられない)**")
        for r in cls:
            print("   条項: %s —— **器具は `tools/empty_clause_apply.py`**" % r["key"])
        print("   **全16便そろった(2026-11-14、7,662頁・312,762層)。空=1 は 74,980層**、")
        print("   **うち 56,777層は名前・種別・列で既に落ちていた** —— 条項の純増は **15,316層**")
        print("   (まだ決まっていなかった層 9,694 + 名前が『除外しない』の行の層 5,622)。")
        print("   **空=1 でも落とさない層 2,090**(画像素材 1,793 / 形式5 149 / 両方 132)")
        print("   **%d + 91,060 + 15,316 = 211,607 / %d = 0.4838**" % (tot, n_all))
        print("   `.lip` の 57,665層と、画素を測れない `.clip` の 58,386層は空を言えない(母数0)")
        print("   **条項2: `image_material`(ラスタライズされていない取り込み画像)** —— 作者の指示で 確定(2026-11-14)。")
        print("   `.clip` 5,517層 —— **見えない 4,356 / 出力に画素が0 267 / 出力に画素あり 821 / 測れない 73**")
        print("   (作者「必ずしも非表示と限らない　出力の範囲外で放置されてるものも多いはず」)")
        print("   + `.lip` だけ 801層。理由は層ごとに4つに名乗るので、危ない理由だけを外せる。")
        print("   **【2026-11-15 棒どおりに理由を1つ撤回した】** 50枚の二択で**作者は 8/50 を「絵の一部」**と答えた")
        print("   (先に登録した棒の 3〜9 の帯)。**理由『出力に画素あり』の 821層を候補に戻す**。")
        print("   うち**7層はカラーの3D背景**(作者の一言)。**棒を読む前 217,124 = 0.4964 →**")
        print("   **理由を外した後 216,303 / 437,425 = 0.4945**(両方に名前を付けて残す。過去の数は撤回しない)")
        print("   (`.lip` の 801 を足すと 217,104 = 0.4963 の見込み)")
        print("   **【2026-11-15 種別 `3d` を空の条項の保留へ】216,303 → 216,301 = 0.4945**(4桁では動かない)")
        print("   **【2026-11-16 (イ) の棒を読んだ。残る2つの理由も撤回】** 盲点だった340層から50枚:")
        print("   **絵の一部 15 / 資料 17 / どれでもない 15 / 分からない 3** —— 「絵の一部」15/50 は")
        print("   **10以上の帯**なので、理由『出力に画素が0』(267)と『測れない』(73)を外す。")
        print("   **棒を読む前 216,301 = 0.4945 → 後 215,961 / 437,425 = 0.4937**(両方を残す)")
        print("   **どれでもない 15 + 分からない 3 = 18 ≥ 10 なので、この群の中身は判定保留**")
        print("   (規則の撤回は確定。『残りが何か』は判じない。『見えない』4,356 へ広げるのは新しい測定)")
        print("   **絵の一部 15 = 3D背景 10 + ラフ・下絵 5**(作者の一言で束ねた。**規則ではない**)。")
        print("   **切り出し・コラージュ 14層は全部『どれでもない』** —— 作者の求めで次の便から")
        print("   ボタンを2つ足す(3D背景 / 切り出し・コラージュ。`config/image_material_choices_20261116.json`)")
        print("   **見えている取り込み画像の層 1,161 は、もう1層も落ちていない。残る理由は『見えない』4,356 だけ**")
        print("   **作者の裁定(2026-11-13、逐語)「下絵　線画hair  blackも名前だけでは落とせない非決定論側だ")
        print("   これらが落とせるのは空の場合だけだ」** —— 『除外しない』の行の 5,622層も、空のときだけ落ちる")
        print("   **【2026-11-23 ベースマスクの裁定を当てた。v8】** 作者が5つの名前を『どの頁でもベースマスク")
        print("   (それ自体に絵を持たない)』と答えた(census の問い)。名前の脚は **105,231 → 110,379(+5,148)**。")
        print("   **ただし 750 層は既に別の脚で落ちていたので純増は 4,398** ——")
        print("   (手元の v3 で脚まで数えると **条項:空(定義C) 750 / 条項:image_material(見えない) 2**。")
        print("   **列 `tone_enabled` と重なった層は 0**。v7 は 750 で、差の2層は版の差)")
        print("   **v7 215,247 = 0.4921 → v8 219,645 / 437,425 = 0.5021**(前後とも残す。過去の版は撤回しない)。")
        print("   手元の v3 で数えると既に落ちていたのは 752 で純増 4,396(**版の差2層**)。")
        print("   `.lip` の 308層は画素が無く、**名前だけが根拠**(画素の裏付けは無い)。")
        print("   `bg base` raster 2,537層は第2回で『だめ』と答えられた層だが、**それは (b)『合成から消す』への答え** ——")
        print("   **a の札は (a)『その層自身の画素が絵か』で、作者が 11-23 に (a) として裁定した。第2回の行は消さない**")
        print("   **【2026-11-24 ユーザーの決定「戻す」。v9】** 名前の裁定で落ちた 4,398層のうち、")
        print("   **v7 で『空だが保留(形式5 を含む。調査中。分析対象外)』だった7層を調査中に戻した** ——")
        print("   『調査中は分析対象外』(作者 10-29)が**名前の裁定より上に来る**(ユーザーの決定 11-24)。")
        print("   **v8 219,645 = 0.50213 → v9 219,638 / 437,425 = 0.50212**(4桁ではどちらも 0.5021。")
        print("   **5桁で区別する**)。名前の脚の純増 4,398 → **4,391**。**v8 は撤回しない**。")
        print("   **`is_base_mask`=1 だが落とさない層が 7 ある** —— 列と判定が食い違う層として名乗る。")
        print("   曝露は動かない: 7層とも土台でなく(`a_is_clipping_base`=0)自身もクリッピングでない(`clip_raw`=0)")
        print("   **【2026-11-26 作者の裁定を当てた。v10】** `下地 N` = **CSP の 3D 素材の LT 変換が作るベースマスク**")
        print("   (作者「使い道がblender のレンダリングで作成されるベースと同じ」)。**4つの名前・161層**(.clip 160 / **.lip 1**)。")
        print("   **v9 219,638 = 0.50212 → v10 219,799 / 437,425 = 0.50248**。161層は全部 a v9 の未決だった。")
        print("   名前の脚 110,379 → **110,540(+161)**。**前後とも残す。過去の版は撤回しない**。")
        print("   **訂正2件(E が自分で書いた誤り)**:")
        print("   (1) E は「161層は全部 raster・`.clip`」と書いたが、**`下地 3` の1層は `.lip`(2014。画素が無い)**。")
        print("   (2) E は「乗る層 0 なので (b) の害は 0」と書いたが、**その `.lip` の1層はクリッピングの土台で乗る層が 1 在る** ——")
        print("   **(b) の害は 0 ではなく 1層。`.clip` の 160層については 0 で正しい**。")
        print("   **名前の総なめ(構造 DB 437,561)で `下地` から始まる層は 164** —— 残る3つ")
        print("   (`下地 1 のコピー`・`下地 2 のコピー`(lip、土台で乗る層 各2)・`下地白 `(clip))は")
        print("   **作者が名を挙げていないので入れていない(規律13)。作者に聞いている**")
        print("   **【2026-11-26 作者の答えで v11】** `下地 1 のコピー`・`下地 2 のコピー`(.lip 各1層)は")
        print("   作者「**同じ扱いでよい**」→ 当てた。**v10 219,799 = 0.50248 → v11 219,801 / 437,425 = 0.50249**")
        print("   (4桁ではどちらも 0.5025。**5桁で区別する**)。名前の脚 110,540 → **110,542**。")
        print("   **`下地白 `(2022b の .clip 1層)は作者「わからない　一応直接見ておく」で保留。入れていない**。")
        print("   **訂正(E が R・Q に出した通達の但し書きを強くしすぎた)**: 「2014 の `.lip` にも LT 変換らしい")
        print("   出力が在る」は**`2014\\cal\\cal.lip` の1頁だけ**の話(下地 3層・輪郭線 4層、計7層が同じ頁)。")
        print("   **1頁では時期の主張にならない**。また census の「.lip の3層は頁は各1」も**同じ1頁**が正しい")
    print("**頁の数は行ごとの和なので、同じ頁を重ねて数えている**(上界)。作品単位は行ごとの最大。")
    print("**残る層: %d。** これは「絵の層」ではなく「この篩で落ちなかった層」である。" % (n_all - tot))
    return 0


CLIP_BASE = "data/census/a_clipping_base_20261123.csv"
VERDICT = "inbox/a_verdict_v3_20261115.csv"


def clipping_base_report(base_path=CLIP_BASE, verdict_path=VERDICT):
    """**この器具は a の判定を1層も動かさない。印字だけ。**

    規律: **a は「その層の画素が絵かどうかの札」で、「合成からその層を消す」ではない**
    (作者 2026-11-23「単に『消していい』という言葉だけで処理すると矛盾がおきる」)。
    クリッピングの土台を合成から消すと、**その上に乗っている層が原稿から消える**。
    だから落とした層のうち土台が何層で、その上に何層乗っているかを**毎回印字する**(規律9)。
    """
    if not (os.path.exists(base_path) and os.path.exists(verdict_path)):
        print("\n(クリッピングの土台の表が無いので、この節は出さない: %s)" % base_path)
        return 0
    print("\n" + "=" * 68)
    print("== クリッピングの土台(a の判定を「削除」に読み替えないための節) ==")
    print("土台の表:", base_path, "sha256", sha(base_path))
    print("判定の表:", verdict_path, "sha256", sha(verdict_path), "**版の名前を書く: v3**")
    V = {}
    with open(verdict_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            V[(r["page_path"], r["layer_id"])] = r.get("a_verdict_v3", "")
    n_base = 0
    agg = collections.defaultdict(lambda: [0, 0, 0])  # 土台 / 乗る層 / うち3Dと分かる
    with open(base_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["a_is_clipping_base"] != "1":
                continue
            n_base += 1
            v = V.get((r["page_path"], r["layer_id"]), "(判定の表に無い)")
            a = agg[v]
            a[0] += 1
            a[1] += int(r["n_clipped_layers"] or 0)
            a[2] += int(r["n_clipped_3d"] or 0)
    print("\n土台: %d 層" % n_base)
    print("| a v3 の判定 | 土台 | 乗る層 | うち 3D と名前で分かる |")
    print("|---|---:|---:|---:|")
    for v, a in sorted(agg.items(), key=lambda x: -x[1][0]):
        print("| %s | %d | %d | %d |" % (v or "(空)", a[0], a[1], a[2]))
    ex = agg.get("template_excluded", [0, 0, 0])
    print("\n**a が既に『絵でない』の札を付けた土台: %d 層。その上に乗っている層: %d 層"
          "(3D と名前で分かるのは %d)**" % (ex[0], ex[1], ex[2]))
    print("**この %d 層は、a を『合成から消す』に読み替えたときに原稿から消える層の上界である。**" % ex[1])
    print("**a の点数は動かない**(札は札のまま)。**乗っている層のうち a が残す層が何層かは、まだ測っていない**")
    print("  —— census に数えてもらう。測るまで『被害は小さい』とも『大きい』とも言わない(判定保留)")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "corpus":
        sys.exit(corpus(*a[1:]))
    if a and a[0] == "clipbase":
        sys.exit(clipping_base_report(*a[1:]))
    rc = main(*(a or []))
    if rc == 0:
        corpus()
        clipping_base_report()
    sys.exit(rc)
