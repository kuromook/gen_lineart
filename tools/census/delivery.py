"""納品の覆い(仕様 §3-4)。**`data/incoming_manuscript/` を読んで、ここで数える。**

**`data/` は読んでよいが、`census/` には1バイトも出さない。**
納品フォルダーと PNG のファイル名には**作品名が入る**ので、段 A では宣言した札に畳む。

**この表の単位は「PNG の枚数」で、「層の数」ではない。**
2026-10-26 に数えて分かったこと —— **何が1枚かが納品ごとに違う**:

- ベタのマスクの納品は **mask だけ**(本体が無い)
- 優先65頁のグラデは **mask だけ 258枚**
- 優先65頁のトーンは **mask 619 と本体 683**(**本体が64枚多い。等しくない**)
- カラー原稿の納品は **本体だけ**で、**頁のプレビューが混ざる**(層ではない)

> **だから「納品 N 枚」は「層 N 枚が届いた」ではない。**
"""
import collections
import os
import re

ROOT_DEFAULT = "/home/sh1/deepl/lineart/data/incoming_manuscript"

# 納品フォルダー → 段 A の札。**作品名を含むものは畳む**(畳んだ数を印字する)
GROUPS = {
    "ako6": ("作品ごとの層の一括納品", True),
    "ako7": ("作品ごとの層の一括納品", True),
    "beta_masks46": ("ベタのマスク", False),
    "beta_masks46_v2": ("ベタのマスク", False),
    "betanuri40_masks": ("ベタのマスク", False),
    "colour_pilot": ("カラー原稿", False),
    "colour_request4_addendum": ("カラー原稿", False),
    "colour_request4_known8": ("カラー原稿", False),
    "colour_request4_rest": ("カラー原稿", False),
    "colour_request5": ("カラー原稿", False),
    "part_layers": ("部分の層", False),
    "priority65_grad_v2": ("優先65頁のグラデ", False),
    "priority65_tone_v2": ("優先65頁のトーン", False),
}

# ファイル名の末尾の項目語 → 段 A の札。**`preview` は層ではない**
ITEMS = {
    "L": "層(番号つき)",
    "beta": "ベタ",
    "tone": "トーン",
    "grad": "グラデーション",
    "komaL": "コマ内の層",
    "lines": "線",
    "preview": "**頁のプレビュー(層ではない)**",
}

NAME = re.compile(r"^(\d+)_(\d{4}[a-z]?)_(.+)\.png$")
TAIL = re.compile(r"_([A-Za-z]+)(\d+)?$")


def scan(root: str, drops):
    """納品の木を歩いて1枚1行にする。**割れなかった名前は理由別に数える**(規律9)。"""
    rows = []
    collapsed = collections.Counter()
    if not os.path.isdir(root):
        drops.drop(f"納品の木が無い: {root}")
        return rows, collapsed
    for d in sorted(os.listdir(root)):
        if d not in GROUPS:
            drops.drop(f"宣言していない納品フォルダー: {d}")
            continue
        group, has_work_name = GROUPS[d]
        if has_work_name:
            collapsed[d] += 1
        for dirpath, _, fns in os.walk(os.path.join(root, d)):
            for fn in fns:
                if not fn.lower().endswith(".png"):
                    continue
                m = NAME.match(fn)
                if not m:
                    drops.drop("PNG の名前が <連番>_<年>_... の形でない", fn)
                    continue
                year, tail = m.group(2), m.group(3)
                is_mask = tail.endswith("_mask")
                core = tail[:-5] if is_mask else tail
                t = TAIL.search(core)
                item_raw = t.group(1) if t else ""
                if item_raw not in ITEMS:
                    drops.drop(f"宣言していない項目語: {item_raw or '(無し)'}", fn)
                    continue
                rows.append(dict(delivery_group=group, delivery_dir=d,
                                 year_label=year, item=ITEMS[item_raw],
                                 is_mask="yes" if is_mask else "no",
                                 page_key=core[:core.rfind("_" + item_raw)] or core))
    return rows, collapsed


def aggregate(rows, keep_private: bool):
    """段 A は納品フォルダー名を持たない(作品名が入るため)。段 C は持つ。"""
    dims = ["delivery_group", "year_label", "item", "is_mask"]
    if keep_private:
        dims.insert(1, "delivery_dir")
    cells = collections.defaultdict(lambda: [0, set()])
    for r in rows:
        key = tuple(r[d] for d in dims)
        cells[key][0] += 1
        cells[key][1].add(r["page_key"])
    out = [list(k) + [v[0], len(v[1])] for k, v in sorted(cells.items())]
    return dims + ["n_png_files", "n_distinct_page_keys"], out


def duplicate_pairs(root: str, rows, keep_private: bool, drops):
    """**納品をまたいで同じ名前の PNG。そして中身が同じかどうか。**

    2026-10-26 に数えて分かったこと —— **同じ名前でも中身は違う。**
    標本20枚の sha256 は「20/20 一致」と出たが、**訂正版と分かっている組
    (壊れた初版の差し替え)を含んでいなかった。** 名指しで全組を比べたら
    **1,636枚のうち 256枚が同名で中身が違った。**

    > **標本は、違うと分かっている組を含まないと何も言えない。**

    **どちらが現行かはここで決めない** —— 版の同定は名前や統計量では決まらない。
    """
    import hashlib

    paths = collections.defaultdict(dict)   # 名前 → {納品フォルダー: 絶対パス}
    for d in sorted(GROUPS):
        base = os.path.join(root, d)
        if not os.path.isdir(base):
            continue
        for dirpath, _, fns in os.walk(base):
            for fn in fns:
                if fn.lower().endswith(".png"):
                    paths[fn][d] = os.path.join(dirpath, fn)

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    cells = collections.Counter()
    n_names = n_dup = 0
    for fn, where in paths.items():
        n_names += 1
        if len(where) < 2:
            continue
        n_dup += 1
        if len(where) > 2:
            drops.flag(f"同じ名前が3つ以上の納品に在る({len(where)}個)", fn)
        ds = sorted(where)
        digests = {d: sha(where[d]) for d in ds}
        for i, a in enumerate(ds):
            for b in ds[i + 1:]:
                if keep_private:
                    key = (a, b)
                else:
                    key = tuple(sorted((GROUPS[a][0], GROUPS[b][0])))
                same = digests[a] == digests[b]
                cells[key + ("same" if same else "different",)] += 1

    dims = (["delivery_dir_a", "delivery_dir_b"] if keep_private
            else ["delivery_group_a", "delivery_group_b"]) + ["content_match"]
    out = [list(k) + [n] for k, n in sorted(cells.items())]
    return dims + ["n_png_names"], out, n_names, n_dup
