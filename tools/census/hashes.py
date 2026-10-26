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
