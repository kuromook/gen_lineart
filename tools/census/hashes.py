"""ファイルの sha256 から、重複を**ここで測る**。

供給元が全 9,653 ファイルの `page_path + sha256` を出したので、
**異なり数と重複の3つの量が、ここで測れる**(それまでは抽出側と Track M の名を帯びていた)。

**この点検が届く範囲**(名前より狭く書く):

- **届く**: 集計の仕方の誤り —— 異なり数、塊の組み方、3つの量の取り違え
- **届かない**: **ハッシュそのもの。** `f_sha256` は抽出側が計算した列で、
  こちらはファイルを開いていない。**抽出側の読みが間違っていれば、同じく間違う**

**重複は3つの別の量である** —— `A − B` は量ではなく手続きで、
引き算で作った数は引き算の意味を名前に持たない。
"""
import collections
import csv


def load(path: str, drops):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            if rec["f_read_error"].strip():
                drops.drop(f"抽出側が読めなかった: {rec['f_read_error'][:40]}")
                continue
            if not rec["f_sha256"].strip():
                drops.drop("sha256 が空")
                continue
            rows.append(dict(page_path=rec["page_path"], fmt=rec["format"],
                             sha=rec["f_sha256"]))
    return rows


# **数のすぐ横に置く述語。** 索引に置くと、数だけが流通する(2026-10-26)。
# **`.lip` の 67.8% は綴じ方についての数で、形式についての数ではない** ——
# これを隣に書かないと `.lip` が「複製の多い形式」として読まれる。
# **固有名は書かない**(段 A は作品名を出さない)
SHARE_PREDICATE = {
    "lip": "**形式の性質ではない。** 双子 1,346 のうち **1,294(96%)は、"
           "同じ中身が別の年のフォルダーに在る組** —— "
           "**作品フォルダーが丸ごと2〜3の年フォルダーに置かれている**"
           "(3つの作品で確認)。"
           "作者の言明と一致する(この時期は運用が安定せず作業途中のコピーが残った)。"
           "**欠陥ではなく仕事の仕方。** 測ったのは抽出側",
    "clip": "",
    "both": "**`.lip` の割合が引き上げている。** 形式ごとの行を見ること",
}

# 塊の大きさに付ける註。**不穏に読める数の横に、平凡な正体を置く**
CLUSTER_NOTE = {
    ("lip", 63): "**21頁 × 3か所。** 中身は**白紙の見開き**(9921×7016)で、"
                 "プレビューはトンボだけ・暗い画素は 0%。"
                 "層の木は揃っており**型紙だけの頁と読める**(抽出側)。"
                 "**画素を全部描いて確かめてはいない。** "
                 "**どれが元でどれが写しかは未定**",
    ("both", 63): "同上(`.lip` の塊)",
}


def quantities(rows):
    """形式ごと・全体の、**3つの別の量**と塊の大きさの分布。

    **`share` の隣に `predicate` を置く** —— 割合が何についての数かを、
    索引ではなく**同じ行に**書く。
    """
    out_q, out_s = [], []
    groups = [("clip", [r for r in rows if r["fmt"] == "clip"]),
              ("lip", [r for r in rows if r["fmt"] == "lip"]),
              ("both", rows)]
    for label, sel in groups:
        if not sel:
            continue
        counts = collections.Counter(r["sha"] for r in sel)
        clusters = {k: v for k, v in counts.items() if v > 1}
        n_files, n_distinct = len(sel), len(counts)
        in_clusters = sum(clusters.values())
        share = f"{len(sel)} の {in_clusters / n_files * 100:.1f}%"
        pred = SHARE_PREDICATE.get(label, "")
        out_q += [
            [label, "n_files", "ファイルの数", n_files, "",
             "**異なり数とは別の量**", ""],
            [label, "n_distinct", "異なり(sha256)", n_distinct, "",
             "**ファイル数とは別の量**", ""],
            [label, "excess", "余り(ファイル − 異なり)", n_files - n_distinct, "",
             "**手続きの結果。複製されたファイルの数ではない**", ""],
            [label, "files_in_clusters", "双子を持つファイルの数", in_clusters, share,
             "**余りでも塊の数でもない**", pred],
            [label, "clusters", "塊の数", len(clusters), "",
             "**ファイルの数ではない**", pred],
        ]
        for size, n in sorted(collections.Counter(clusters.values()).items()):
            out_s.append([label, size, n, CLUSTER_NOTE.get((label, size), "")])
    return out_q, out_s


def grad_layer_map(path: str, drops):
    """グラデ v1/v2 を**層を鍵に**突き合わせた表(供給元の対応表から数え直す)。

    名前を鍵にすると 152枚が「違う」と出るが、`_grad{N}` の `N` は頁の中の順番で、
    **v2 で層が加わって後ろにずれた。** 層を鍵にすると分かれ方が変わる。
    """
    cat = collections.Counter()
    moved = both_cover = 0
    n = 0
    with open(path, encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            n += 1
            same_name = rec["same_name"] == "1"
            ident = rec["png_identical"] == "1"
            if not rec["v1_file"].strip():
                cat["v2 にしか無い層(v1 に無い)"] += 1
            elif ident and same_name:
                cat["画素も名前も同じ"] += 1
            elif ident:
                cat["画素は同じ・名前が違う(見かけの差)"] += 1
            elif same_name:
                cat["画素が違う・名前は同じ(本当の変更)"] += 1
            else:
                cat["画素も名前も違う(本当の変更)"] += 1
            a, b = rec["v1_mask_cover"].strip(), rec["v2_mask_cover"].strip()
            if a and b:
                both_cover += 1
                if float(a) != float(b):
                    moved += 1
    rows = [[k, v] for k, v in sorted(cat.items(), key=lambda x: -x[1])]
    return rows, n, both_cover, moved


# **畳む規則は作者の裁定である**(2026-10-28)——
# バイト同一の組は**より古い札の写しを代表に残す**。
#
# **2026-10-28 に一度逆に書いて公開した。取り下げた誤りである** ——
# 「**年代が新しい方を残す**」は **Track M の便にあった記述**で、
# **こちらは作者に直接確かめずにそれを前提にした。**
# 作者の言明は「**ページが同バイト数なら年代の古いものを代表とする**」。
# **こちらが作者から直接もらっていたのは「どちらが新しいか」だけで、
# 「どちらを残すか」は聞いていなかった**(T146 の家族: 借りた前提を確かめない)。
#
# 札の前後そのものは作者の裁定どおり: **同じ暦年では `b`/`c` の付いた札が新しい**
# (`2024` < `2024b` < `2024c`、`2022` < `2022b`、`2019` < `2019b`)。
# **残すのはその前後の「古い側」である。**
#
# **札そのものは畳まない**(`2019` と `2019b` は別の池。作者の決定 2026-10-26)——
# 畳むのは**代表の選択だけ**で、残った1枚はその札の行に数える。
#
# **この決め方が年ごとの数を入れ替える。** 同じ暦年で札が違う組が 527 あり、
# 綴りの前後をひっくり返すと **517頁が札の間を動く**。
# **合計 7,418 はどちらの読みでも動かないので、合計の突き合わせでは映らない**
# (2026-10-28、基盤との突き合わせで 11量が一致した後に見つかった)。
def _newer(label: str):
    """2つの札のどちらが新しいかの順序。**暦年が先、同じなら綴りの後ろが新しい。**

    **この関数は「新しさ」だけを決める。どちらを残すかは呼ぶ側が決める** ——
    代表は**古い側**(`min`)である(作者の裁定 2026-10-28)。
    """
    return (label[:4], label[4:])


def folded_by_year(rows, year_of, drops):
    """**年 × 形式ごとに、畳む前と畳んだ後の頁数。**

    返すのは `{(year_label, fmt): (畳む前, 畳んだ後)}` と、決め方の内訳。

    **`.clip` と `.lip` は同じバイト列になり得ない**(容れ物が違う)ので、
    組は形式の中で閉じる —— だから形式ごとに数えても組が割れない。
    **それでも形式を跨いで組を作って数え、跨いだ組が 0 であることを印字する**
    (**閉じていることを仮定ではなく数で言う**)。
    """
    clusters = collections.defaultdict(list)
    for r in rows:
        yr = year_of(r["page_path"])
        if not yr:
            drops.drop("年の札が `\\d{4}[a-z]?` の形でない")
            continue
        clusters[r["sha"]].append((yr, r["fmt"]))

    before = collections.Counter()
    folded = collections.Counter()
    aux = dict(n_clusters=0, n_multi_year=0, n_tie_label=0, n_cross_format=0,
               tie_pairs=collections.Counter())
    for sha, mem in clusters.items():
        aux["n_clusters"] += 1
        for yr, fmt in mem:
            before[(yr, fmt)] += 1
        if len({f for _, f in mem}) > 1:
            aux["n_cross_format"] += 1
        if len({yr[:4] for yr, _ in mem}) > 1:
            aux["n_multi_year"] += 1
        # **代表は最も古い札の側**(作者の裁定 2026-10-28)
        oldest = min(_newer(yr) for yr, _ in mem)
        top = sorted({(yr, fmt) for yr, fmt in mem if _newer(yr) == oldest})
        if len({yr for yr, _ in mem if _newer(yr)[0] == oldest[0]}) > 1:
            aux["n_tie_label"] += 1
            aux["tie_pairs"][tuple(sorted({yr for yr, _ in mem
                                           if yr[:4] == oldest[0]}))] += 1
        folded[top[0]] += 1
    return {k: (before[k], folded.get(k, 0)) for k in before}, aux


def period_by_year(rows, year_of, drops):
    """**「最も早く現れた札」**を札ごとに数える(Track M から渡された列、2026-10-28)。

    **2026-10-28 の裁定で、代表の札と同じ量になった** ——
    作者は「**ページが同バイト数なら年代の古いものを代表とする**」と決めたので、
    **代表の札 = 最も古い札 = この列**である。
    **だからここでは「採らなかった読み」(最も新しい札)と並べて置く** ——
    規則が逆だったときの数を**取り下げた記録として残す**ためで、
    **公開しているのは代表の側(最も古い札)である。**

    **この列が答えないこと(Track M と基盤から、列と一緒に受け取った限界)**:

    - **「制作時期」ではない。** 年代札はどの頁についても**「それ以降ではない」という上限**で、
      **早い札に付け替えられるのは、同じバイト列が早いフォルダにも在る頁だけ**
    - **1回しか現れない頁は、使い回されていても補正されない**
    - **補正は不均等で、不均等自身が傾向を作り得る**(複製の多い年ほど頁を早い年へ差し出す)
    - **走行の主値の層別にこの札を使わない**(基盤の制約)—— 年別の差が補正の不均等を映す

    **裁定の前は、この量を公開物に出さない判断だった**(2026-10-28、ユーザー)——
    **列の名前だけ見ると制作が早まったように読めるのに、補正が不均等**だからである。
    **裁定で公開列がこの量になったので、その限界は公開側の但し書きに移した。**
    この表は段 C に残す(**2つの読みを並べた記録**)。
    """
    clusters = collections.defaultdict(list)
    for r in rows:
        yr = year_of(r["page_path"])
        if not yr:
            drops.drop("年の札が `\\d{4}[a-z]?` の形でない")
            continue
        clusters[r["sha"]].append(yr)
    period, repr_ = collections.Counter(), collections.Counter()
    n_multi = n_diff = 0
    # **「同順位」は3つの別の量である**(2026-10-28、Track M との突き合わせで分かれた)——
    # **同じ暦年の札の前後(`2024` < `2024b` < `2024c`)が、2つの列に別の大きさで効く**:
    # **代表の札(= 最も古い札。作者の裁定)に効くのは群の最も古い暦年の同順位で、
    # 取り下げた読み(新しい方を残す)に効くのは最も新しい暦年の同順位。**
    # **どこかの暦年の同順位**(その和集合)はどちらの分母でもない。
    # **合計も門も一致したまま、分母だけが違う** —— T146 の向きを逆にした形
    n_tie_newest = n_tie_oldest = n_tie_any = 0
    for mem in clusters.values():
        early, late = min(mem, key=_newer), max(mem, key=_newer)
        period[early] += 1
        repr_[late] += 1
        labels = set(mem)
        if len(labels) > 1:
            n_multi += 1
        if early != late:
            n_diff += 1
        years = collections.Counter(l[:4] for l in labels)
        if years[max(years)] > 1:
            n_tie_newest += 1
        if years[min(years)] > 1:
            n_tie_oldest += 1
        if any(v > 1 for v in years.values()):
            n_tie_any += 1
    return period, repr_, dict(n_clusters=len(clusters), n_multi_label=n_multi,
                               n_period_differs=n_diff,
                               n_tie_newest_year=n_tie_newest,
                               n_tie_oldest_year=n_tie_oldest,
                               n_tie_any_year=n_tie_any)
