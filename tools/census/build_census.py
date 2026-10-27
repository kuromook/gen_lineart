#!/usr/bin/env python
"""原稿の国勢調査を1本の走行で組む。

**1本の器具が、同じ集計から公開(段 A)と全量(段 C)を同時に出す**(`doc/census_spec.md` §2)。
2本に分けると「公開版だけ古い」が起きる。

段 A(`census/`、**追跡される = 全世界から読める**)に出るのは
**年・形式・種別ごとの件数と分布だけ。固有名も別名も行の鍵も出さない。**
段 C(`data/census_full/`、追跡されない)は作品・頁まで持つ。

公開側の行は **PUBLIC_DIMS だけを持つ鍵から組み立てる** —— 固有名を落とすのではなく、
**名前の欄を一度も持たない**(落とす実装は落とし忘れを静かに通す。T104)。

使い方:
    ./venv/bin/python tools/census/build_census.py
    ./venv/bin/python tools/census/build_census.py --public-out census --full-out data/census_full
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caveats import CAVEATS  # noqa: E402
import delivery  # noqa: E402
import full_population as fp  # noqa: E402
import faint  # noqa: E402
import hashes  # noqa: E402
import occluders  # noqa: E402
import sources  # noqa: E402

# 公開側の行が持てる次元。**ここに名前を足してはいけない**
PUBLIC_DIMS = ("population", "year_label", "v_layer_kind", "layer_type")
# 全量側だけが持つ次元
PRIVATE_DIMS = ("work", "subdir", "page_path")

# 年の区画名は公開する(段 A は「年ごとの件数」を許す)。
# **作者が答えた(2026-10-26): `2022b` は暦年 2022 の続きである。**
# だから暦年は `calendar_year` の列で出す。**それでも行は畳まない** ——
# 畳まない根拠は「暦年が分からないから」ではなく **世代もプールである**(別の折り返しは
# 別のプール)。**作者の答は前者を解消したが、後者は変えない**
ALLOWED_PUBLIC_TOKENS = set()


def calendar_year(year_label: str) -> str:
    """フォルダー名から暦年を取る。`2022b` → `2022`。

    **作者の言明**(2026-10-26): **`2022b` は `2022` の続き。**
    **`2019b` `2020b` `2023b` `2024b` `2024c` にも同じ綴りの規則を当てている** ——
    **そこは作者に未確認**(規律6: 作者の判断が正本)。
    """
    head = year_label[:4]
    return head if head.isdigit() else ""

# **母集団ごとに、その材料が実際に運んでいる量。**
# ここに無い量は **0 ではなく空欄**で出す —— 「測っていない」を 0 と書くと、
# 読み手は「測って 0 だった」と読む(この台帳が何度も払った代償)
POPULATION_METRICS = {
    "req3_layer_kind": ("n_pages", "n_pages_without_work", "n_works",
                        "ext_clip", "ext_lip"),
    "added33_structure": ("n_pages", "n_pages_without_work", "n_works",
                          "ext_clip", "ext_lip",
                          "sum_n_layers_incl_folders", "sum_n_folders",
                          "sum_n_text_layers", "sum_n_image_material_layers"),
}

# 公開してよい次元の値(**白名簿。** 宣言外の値は点検が指摘する)
ALLOWED_POPULATIONS = ("req3_layer_kind", "added33_structure",
                       "db_files", "db_distinct", "db_layers", "lip_files",
                       "db_all_layers", "body_initcolor", "priority65",
                       "delivery_png", "clip_files", "grad_v1_v2")
ALLOWED_LAYER_KINDS = ("raster", "mask_fill", "image_material", "other", "text",
                       "paper", "story_info", "filter", "gradation", "vector", "3d")


def pop(key: str) -> str:
    """`population` の列に書く値は、**必ずここを通す。**

    **同じ誤りを3回したなら、それは不注意ではなく器具の形である**(2026-10-26)——
    `population` の列に述語の散文を書くのを、1日に3回やった(全数の表 / 重複の量 /
    グラデ)。3回とも点検が commit の前に止め、直し方も3回同じだった。

    **止める器具が在ったので費用は各回1行で済んだが、止まるのは走行の最後である。**
    だから**書けなくする** —— 宣言していない鍵は、ここで即座に落ちる。

    **述語は `populations.csv` に置く。** `population` の列は鍵だけを運ぶ。
    """
    if key not in ALLOWED_POPULATIONS:
        raise SystemExit(
            f"`population` の列に宣言外の値: {key!r}\n"
            f"  **鍵を書くこと。述語は `populations.csv` に置く。**\n"
            f"  宣言済みの鍵: {', '.join(ALLOWED_POPULATIONS)}")
    return key

# **直る予定のある列は、集計に固定しない**(基盤 2026-10-26)。
# 「いま測れる」と「いま持つべき」は別。抑えた行は**数えて印字し、空欄で出す**(0 にしない)
SUPPRESSED = {
    ("drawn_ratio", "mask_fill"):
        "本体は InitColor 255 = 全面。11,455層が 0 と読まれている。"
        "抽出側が読み手を直すと値が動く(T129)",
    ("raster_cover", "*"):
        "grad の96層が 0.0 と出ているが InitColor どおりなら全面。"
        "抽出側が出し直す。**この列は集計に入れない**",
}
# **層の名前も段 A には出さない**(仕様 §2)。ところが禁止語彙は `page_path` から
# 組むので、**層名を1つも見ていない。** 便で名前を知った分をここに宣言して塞ぐ。
# **これは語彙の穴を塞ぐだけで、知らない層名には届かない**(届く範囲を名前より狭く書く)
# **作者が付けた層名。出さない。**
AUTHOR_LAYER_NAMES = (
    # **`線画` 単独は入れない** —— このプロジェクトの題材そのものの語で、
    # 一般名詞として散文に出る(2026-10-27 に自分の但し書きで鳴った)。
    # 漏れるのは**複合した層名**なので、そちらを禁止する
    "線画 hair", "肌グラデ", "乳首グラデ", "モザイク", "差分マスク", "正面",
)
# **アプリが付ける層名。出してよい** —— 作者の作業名ではなく、形式の部品の名前。
# これを禁止すると `layer_type` のビットの根拠が書けなくなる(根拠の無い表は持てない)
SYSTEM_LAYER_NAMES = {
    "ノンブル": "CLIP の頁番号層。アプリが付ける名前",
    "作品情報": "CLIP の作品情報層。アプリが付ける名前",
    "用紙": "CLIP の用紙層。1頁に1個。アプリが付ける名前",
    "選択範囲": "CLIP の選択範囲層。アプリが付ける名前",
    "コマ": "コマ枠。形式の部品の名前(作者の作業名と重なりうるが、"
            "ここで出るのは枠の種類としての語)",
}

# 散文に出てよい語。**語ごとに理由を書く。** 理由の無い除外を作らない
PROSE_EXCEPTIONS = {
    "clip": "ファイルの拡張子(`.clip`)。作品名ではない",
    "lip":  "ファイルの拡張子(`.lip`)。作品名ではない",
    "work": "英単語として使う(「作品が無い」の意)。同名のフォルダーが在るのは偶然",
    "copy": "英単語。同名のフォルダーが在るのは偶然",
    "tmp":  "英単語。同名のフォルダーが在るのは偶然",
    "top":  "英単語。同名のフォルダーが在るのは偶然",
    "task": "英単語。同名のフォルダーが在るのは偶然",
    "done": "英単語。同名のフォルダーが在るのは偶然",
    "color": "英単語・CSS の語。同名のフォルダーが在るのは偶然",
    "cover": "英単語・CSS の語。同名のフォルダーが在るのは偶然",
    "線画": "**この題材そのものの一般名詞。** 禁止すると但し書きが書けなくなる。"
            "漏れるのは `線画 hair` のような**複合した層名**なので、そちらを禁止する",
}

# `drawn_ratio` をその種別で読んでよいか。**読めない列を空欄で出し、0 にしない**
DRAWN_RATIO_USABLE = {
    "mask_fill": ("no", "本体は InitColor 255 = 全面。0 は空を意味しない(T129)。近日値が動く"),
    "image_material": ("no", "絵は ResizableOriginalMipmap。読み手は別の所を見ていた(T129)"),
}
DRAWN_RATIO_USABLE_DEFAULT = ("unverified", "この種別については誰もまだ確かめていない")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def instrument_version() -> dict:
    """**この数を計算した器具の版。** 出力に刻む。

    **器具を直したら、直す前に取った数を全部捨てる** —— だが
    **出力は再生成されるが、文章は再生成されない**(2026-10-27 に踏んだ。T139)。
    点検は全部通り、成果物も正しく、**報告だけが単独で誤っていられる** ——
    走行の中に検出手段が無い。

    **だから防げない。ただし「古い」と分かる形にはできる** ——
    **報告に、その数を計算した器具の版を書く**(抽出側が納品通達で既にやっている作法)。
    **作法のままにせず、走行が自分で刻む。**

    **`dirty` が yes のときは、版が器具を同定しない** —— そう書く。
    """
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    head = git("rev-parse", "--short", "HEAD")
    # **器具のファイルだけを見る** —— 公開物の差分で dirty になっても版は同定できる
    changed = git("status", "--porcelain", "--", "tools/census", "tools")
    return dict(commit=head or "(git が読めない)",
                dirty="yes" if changed else "no",
                dirty_files=len([x for x in changed.splitlines() if x.strip()]))


def today() -> str:
    """見出しの時刻は `date` から機械的に差し込む。推定で書かない。"""
    return subprocess.run(["date", "+%Y-%m-%d"], capture_output=True, text=True,
                          check=True).stdout.strip()


class Drops:
    """**選別は測定である。落としたものを理由別に数えて印字する**(規律9)。

    `continue` の前にここを通す。数えない `continue` を書かない。
    **落としたもの(`drop`)と、残したが注記の付く行(`flag`)を別に数える** ——
    混ぜると「落ちていない行」が落ちたように読める。
    """

    def __init__(self):
        self.dropped = collections.Counter()
        self.flagged = collections.Counter()
        self.examples = {}

    def drop(self, reason: str, what: str = "") -> None:
        self.dropped[reason] += 1
        self.examples.setdefault(reason, what)

    def flag(self, reason: str, what: str = "") -> None:
        self.flagged[reason] += 1
        self.examples.setdefault(reason, what)

    def report(self, label: str) -> None:
        for title, counter in (("落とした行", self.dropped),
                               ("残したが注記の付く行", self.flagged)):
            print(f"  [{title}: {label}]")
            if not counter:
                print("    なし(0件)")
                continue
            for reason, n in counter.most_common():
                print(f"    {n:7d}  {reason}")


def parse_page_path(page_path: str, drops: Drops):
    """`F:\\manuscripts\\<年>\\[作品\\[下位...\\]]<ファイル>` を割る。

    **割れなかったものは落とさずに返す** —— 落とすと母集団が黙って縮む。
    戻り値の `year_label` が None の行は、呼び手が理由付きで数える。
    """
    segs = page_path.split("\\")
    if len(segs) < 4 or segs[1].lower() != "manuscripts":
        drops.drop("page_path が F:\\manuscripts\\... の形でない", page_path)
        return None
    year_label = segs[2]
    fname = segs[-1]
    ext = os.path.splitext(fname)[1].lower()
    if len(segs) == 4:
        # 年フォルダー直下のファイル。**作品が「無い」(0 ではない)**
        work, subdir = "", ""
        has_work = False
    else:
        work, subdir = segs[3], "\\".join(segs[4:-1])
        has_work = True
    return dict(year_label=year_label, work=work, subdir=subdir,
                page_path=page_path, ext=ext, has_work=has_work)


def load_req3(path: str):
    """依頼3の層ごとの行。**フォルダー層は入っていない**(供給元の明記)。"""
    drops = Drops()
    rows = []
    pages = {}
    with open(path, encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            parsed = parse_page_path(rec["page_path"], drops)
            if parsed is None:
                continue
            pages.setdefault(rec["page_path"], parsed)
            raw = rec["drawn_ratio"].strip()
            if raw == "":
                bucket = "blank"          # **0 ではない。** rendered=false の層
                drops.flag("drawn_ratio が空欄 → n_drawn_blank に数える"
                           "(**落としていない。0 とは別の桶**)", rec["page_path"])
            else:
                v = float(raw)
                bucket = "gt0" if v > 0 else "zero"
            rows.append(dict(parsed, layer_id=rec["layer_id"],
                             v_layer_kind=rec["v_layer_kind"],
                             layer_type=rec["layer_type"],
                             has_image_material=rec["has_image_material"],
                             rendered=rec["rendered"], drawn_bucket=bucket))
    return rows, pages, drops


def load_added33(path: str):
    """後から入った33頁の構造。**層数はフォルダーを含む**(依頼3と比べない)。"""
    drops = Drops()
    rows = []
    with open(path, encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            parsed = parse_page_path(rec["page_path"], drops)
            if parsed is None:
                continue
            if rec["year"] and rec["year"] != parsed["year_label"]:
                drops.flag("供給元の year 列と page_path の年区画が食い違う"
                           "(**落とさない。** 食い違いを数えて出す)", rec["page_path"])
            rows.append(dict(parsed,
                             n_layers=int(rec["n_layers"]),
                             n_folders=int(rec["n_folders"]),
                             n_text_layers=int(rec["n_text_layers"]),
                             n_image_material_layers=int(rec["n_image_material_layers"]),
                             canvas_w=rec["canvas_w"], canvas_h=rec["canvas_h"],
                             dpi=rec["dpi"]))
    return rows, drops


def load_initcolor(path: str):
    """本体の `InitColor` の納品(**1層1行**)。だから**ここで測れる** ——
    要約を引用するのではなく、行から数え直す(T132 の避け方)。

    `set` 列には **`fill_ako6` のように作品名が入る**ので、
    **段 A では種類(トーン / グラデーション / ベタ)に畳む** ——
    畳んだ世代の数を印字する(**畳んだ行は、畳んだと分かる形で出す**)。
    """
    drops = Drops()
    rows = []
    collapsed = collections.Counter()
    with open(path, encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            if rec["error"].strip():
                drops.drop(f"抽出側が error を付けた行: {rec['error'][:40]}")
                continue
            raw_set = rec["set"]
            if raw_set in fp.SET_LABELS:
                kind = fp.SET_LABELS[raw_set]
            elif raw_set.startswith(fp.FILL_PREFIX):
                kind = fp.FILL_LABEL
                collapsed[raw_set] += 1
            else:
                drops.drop(f"宣言していない set: {raw_set}")
                continue
            def num(key):
                v = rec[key].strip()
                return None if v == "" else float(v)
            stored = int(rec["stored"] or 0)
            blocks = int(rec["blocks"] or 0)
            rc = num("raster_cover_in_manifest")
            amax = num("stored_alpha_max")
            rows.append(dict(kind=kind, init=rec["init"].strip(),
                             bound_class=classify_bound(rec["init"].strip(),
                                                        blocks, stored),
                             layer_type=rec["layer_type"].strip(),
                             has_stored_blocks="yes" if stored else "no",
                             rc_bucket=("manifest に無い" if rc is None
                                        else "0.0" if rc == 0 else ">0"),
                             alpha_max_zero=("yes" if (amax is not None and amax == 0)
                                             else "no" if amax is not None else ""),
                             ))
    return rows, drops, collapsed


# **読み手は保存されていないブロックを 0 で埋める**(抽出側 2026-10-26)。
# そのため **`body_init == 255` かつ一部保存の層では、被覆の値が下限になる。**
# **0 は特別ではなく、下限が 0 まで下がった場合でしかない。**
BOUND_CLASSES = (
    ("255 + 保存0", "全面の可能性", ""),
    ("0 + 保存0", "空", ""),
    ("0 + 保存あり", "中身を見る(埋める 0 が InitColor と同じなので偏らない)", ""),
    ("255 + 全部保存", "**値が値どおり**", ""),
    ("255 + 一部保存", "**下限**", "平均を出さない / 順位を付けない / "
                               "下限の向きに閾を置かない"),
)


def classify_bound(init: str, blocks: int, stored: int) -> str:
    """被覆の値が、値か下限か、全面の可能性か。**5つに分ける**(抽出側の T60 拡張)。"""
    if init == "255":
        if stored == 0:
            return "255 + 保存0"
        if blocks and stored == blocks:
            return "255 + 全部保存"
        return "255 + 一部保存"
    return "0 + 保存0" if stored == 0 else "0 + 保存あり"


def emit_req3_bound_classes(outdir: str, faint_rows, req3_pages):
    """**依頼3の母集団(56,576層)で、被覆が下限の層を計数する。**

    それまで「機序は確定・**件数は未計数**」と書いていた —— `blocks`/`stored` が
    無かったため。**2026-10-27 の便で埋まった。**
    """
    cells, pages, types, rendered = faint.bound_classes(
        faint_rows, classify_bound, set(req3_pages))
    meaning = {c: m for c, m, _ in BOUND_CLASSES}
    rows = []
    for cls, _, _ in BOUND_CLASSES:
        n = cells.get(cls, 0)
        drawing = sum(v for k, v in types[cls].items()
                      if k in faint.DRAWING_LAYER_TYPES)
        fill = sum(v for k, v in types[cls].items()
                   if k in faint.LAYER_TYPE_EQ2)
        rows.append([cls, meaning.get(cls, ""), n, len(pages.get(cls, ())),
                     rendered.get(cls, 0), n - rendered.get(cls, 0),
                     drawing, fill, n - drawing - fill])  # 軸は layer_type
    written = write_csv(os.path.join(outdir, "req3_bound_classes.csv"),
                        ["bound_class", "meaning", "n_layers", "n_pages",
                         "n_rendered", "n_not_rendered",
                         # **軸を名前に持っていく**(基盤 2026-10-27 の第3の形:
                         # 型で数えて種類の名前で印字した)。
                         # `layer_type & 1` = 本体がラスタ / `layer_type == 2` =
                         # **マスクのみでラスタ本体が無い層** —— これは
                         # **`v_layer_kind = mask_fill` と同じ集合ではない**
                         # (全数で型2が 119,619、うち塗り 118,370・画像素材 799・
                         #  ベクター 437・テキスト 13)
                         "n_layer_type_bit1", "n_layer_type_eq2",
                         "n_layer_type_other"], rows)
    print("  [被覆の分類: 依頼3の 56,576層] **件数が計数できた**(未計数が外れた)")
    for r in rows:
        print(f"      {r[0]:14s} 層 {r[2]:6d} / 頁 {r[3]:4d} / "
              f"type&1 {r[6]:5d} / type==2 {r[7]:5d} / rendered {r[4]:6d}")
    print("    **下限(分類5)は 512層だが、値が在るのは 218層。** "
          "294層は `drawn_ratio` が空欄で**下限にすらならない**")
    print("    **危険は消えたのではなく階級が移った** —— "
          "**分類1(255 + 保存0 = 全面の可能性)が 11,499層**で、"
          "**その `drawn_ratio` は全部 0。** "
          "ほとんどが **`layer_type == 2`(マスクのみ・ラスタ本体なし)**")
    return written


def emit_faint_gap(outdir: str, faint_rows, req3_pages):
    """**`drawn_ratio` の閾 26 が落としている薄い画素。**

    `alpha_ge1_ratio` との差が、落ちている量。**描画の層で落ちているなら薄い線である。**
    """
    out, skipped = faint.faint_gap(faint_rows, set(req3_pages))
    rows = [[d, b, n] for d, b, n in out]
    written = write_csv(os.path.join(outdir, "faint_alpha_gap.csv"),
                        ["is_drawing_layer", "gap_bucket", "n_layers"], rows)
    print("  [薄い画素] `drawn_ratio`(アルファ≥26)が アルファ≥1 から落ちている量")
    for d, b, n in out:
        print(f"      描画層={d:3s} {b:16s} {n:6d}")
    for why, n in skipped.items():
        print(f"      飛ばした {n}: {why}")
    return written


def emit_bound_classes(outdir: str, ic_rows, keep_private: bool):
    """**値が下限である層が何層か。** ここで測れる範囲だけ。

    **測れるのは本体の `InitColor` が届いた 2,095層だけ** ——
    依頼3の 56,576層には `blocks`/`stored` が無いので、
    **そちらの該当数は分からない**(0 ではなく、不明)。
    """
    key = ("kind", "bound_class")
    agg = collections.Counter(tuple(r[k] for k in key) for r in ic_rows)
    meaning = {c: m for c, m, _ in BOUND_CLASSES}
    rule = {c: r for c, _, r in BOUND_CLASSES}
    rows = []
    for (kind, cls), n in sorted(agg.items()):
        rows.append([kind, cls, meaning.get(cls, ""), n, rule.get(cls, "")])
    # **出なかった分類も 0 で印字する** —— 無い行は「測っていない」と区別が付かない
    seen = {c for _, c in agg}
    for c, m, r in BOUND_CLASSES:
        if c not in seen:
            rows.append(["(どの種類にも無し)", c, m, 0, r])
    return write_csv(os.path.join(outdir, "coverage_bound_classes.csv"),
                     ["kind", "bound_class", "meaning", "n_layers", "rule"], rows)


def emit_initcolor(outdir: str, rows, keep_private: bool):
    """種類 × `InitColor` × 保存ブロック × 被覆の桶。**段 A も段 C も同じ鍵**
    (この表に頁の鍵は無い。`set` を畳んだだけが段の差)。"""
    key = ("kind", "init", "layer_type", "has_stored_blocks", "rc_bucket", "alpha_max_zero")
    agg = collections.Counter(tuple(r[k] for k in key) for r in rows)
    out = [list(k) + [n] for k, n in sorted(agg.items())]
    return write_csv(os.path.join(outdir, "body_initcolor.csv"),
                     list(key) + ["n_layers"], out)


def emit_canvas(outdir: str, a33_rows):
    """画布の宣言値の分布。**単位は行ごとに決められない**(T126)。

    `pages_added33_structure_20261026.csv` には **`canvas_unit` 列が無い** ので、
    497×277 や 210×297(mm と読める)と 3000×3000 や 2033×3433(px と読める)が
    同じ列に並ぶ。**どちらかをここで決めない** —— `v_canvas_px_w/h` を待つ。
    """
    cells = collections.Counter(
        (r["canvas_w"], r["canvas_h"], r["dpi"]) for r in a33_rows)
    rows = [[w, h, d, n, "no",
             "`canvas_unit` 列が無い。mm と読める行と px と読める行が同じ列に並ぶ(T126)"]
            for (w, h, d), n in sorted(cells.items(), key=lambda x: -x[1])]
    return write_csv(os.path.join(outdir, "canvas_declared.csv"),
                     ["canvas_w", "canvas_h", "dpi", "n_pages",
                      "canvas_unit_resolved", "why"], rows)


def emit_full_population(outdir: str):
    """全数 437,561層の表。**供給元が測った。ここでは測れない**(DB が手元に無い)。"""
    written = {}
    written["layer_type_bits.csv"] = write_csv(
        os.path.join(outdir, "layer_type_bits.csv"),
        ["bit", "meaning", "confidence", "evidence", "not_answers",
         "population", "measured_by", "as_of"],
        [[b["bit"], b["meaning"], b["confidence"], b["evidence"], b["not_answers"],
          pop("db_all_layers"), fp.SUPPLIER, fp.AS_OF] for b in fp.BITS])
    written["layer_type_values.csv"] = write_csv(
        os.path.join(outdir, "layer_type_values.csv"),
        ["layer_type", "bits", "is_folder", "n_layers", "content",
         "population", "measured_by", "as_of"],
        [[v["layer_type"], v["bits"], v["is_folder"], v["n_layers"], v["content"],
          pop("db_all_layers"), fp.SUPPLIER, fp.AS_OF] for v in fp.VALUES])
    written["mask_by_kind.csv"] = write_csv(
        os.path.join(outdir, "mask_by_kind.csv"),
        ["is_folder", "has_frame_line", "layer_type_and_2", "has_mask", "n_layers",
         "population", "measured_by", "as_of"],
        [[m["is_folder"], m["has_frame_line"], m["layer_type_and_2"], m["has_mask"],
          m["n_layers"], pop("db_all_layers"), fp.SUPPLIER, fp.AS_OF]
         for m in fp.MASK_BY_KIND])

    written["visibility_exceptions.csv"] = write_csv(
        os.path.join(outdir, "visibility_exceptions.csv"),
        ["population", "direction", "n_layers", "detail", "measured_by", "as_of"],
        [[pop(v["population"]), v["direction"], v["n_layers"], v["detail"],
          v["measured_by"], v["as_of"]] for v in fp.VISIBILITY_EXCEPTIONS])
    return written


def emit_delivery(outdir: str, root: str, keep_private: bool, drops):
    """納品の覆い(仕様 §3-4)。**`data/` を読んで数え、`census/` には札だけ出す。**"""
    rows, collapsed = delivery.scan(root, drops)
    written = {}
    header, out = delivery.aggregate(rows, keep_private)
    written["delivery_coverage.csv"] = write_csv(
        os.path.join(outdir, "delivery_coverage.csv"), header, out)
    header, out, n_names, n_dup = delivery.duplicate_pairs(root, rows, keep_private, drops)
    written["delivery_duplicates.csv"] = write_csv(
        os.path.join(outdir, "delivery_duplicates.csv"), header, out)
    header, out = delivery.duplicate_by_content(root, keep_private, drops)
    written["delivery_duplicates_by_content.csv"] = write_csv(
        os.path.join(outdir, "delivery_duplicates_by_content.csv"), header, out)
    return written, rows, collapsed, n_names, n_dup


def emit_duplicates(outdir: str, hash_rows, grad):
    """重複の3つの量と塊の大きさ。**ここで測った**(供給元の名は外れた)。"""
    written = {}
    q, sizes = hashes.quantities(hash_rows)
    written["duplicate_quantities.csv"] = write_csv(
        os.path.join(outdir, "duplicate_quantities.csv"),
        ["format", "quantity", "label", "n", "share", "not_what", "predicate",
         "measured_by", "as_of"],
        [r + ["census track(sha256 の列から数えた)", fp.AS_OF] for r in q])
    written["cluster_sizes.csv"] = write_csv(
        os.path.join(outdir, "cluster_sizes.csv"),
        ["format", "cluster_size", "n_clusters", "note", "measured_by", "as_of"],
        [r + ["census track(sha256 の列から数えた)", fp.AS_OF] for r in sizes])
    grad_rows, n_layers, both_cover, moved = grad
    written["grad_v1_v2_by_layer.csv"] = write_csv(
        os.path.join(outdir, "grad_v1_v2_by_layer.csv"),
        ["category", "n_layers", "population", "measured_by", "as_of"],
        [r + [pop("grad_v1_v2"),
              "census track(供給元の対応表から数え直した)", fp.AS_OF]
         for r in grad_rows])
    return written


# **`populations.csv` に直に書く行の `needs` と、その満たされ方。**
# **散文を語で当てない** —— 2026-10-26 に「`needs` に『層数』と在って `n_layers` が
# 埋まっていれば古い」という当て方をして、**`db_layers` で偽陽性を出した**
# (あの `needs` は「値が無い」ではなく「**同じ走行で取り直したい**」だった)。
# **点検の文言は発火した条件から導く**(T134 (h))—— だから条件を宣言で持つ。
#
#   ("column", <列>)  その列が埋まれば満たされる
#   ("source", <鍵>)  その材料が届けば満たされる
#   None              **この行の needs は、値が埋まっても満たされない**(取り直しの依頼)
INLINE_NEEDS = {
    # **依頼した材料は、必ずここに行として在る** ——
    # 2026-10-26 に `blocks`/`stored` を依頼したのに `needs` に行が無く、
    # **`canvas_unit` は `needs` に在るのに依頼に入れていなかった** ——
    # **依頼と表が両方向にずれた。** 表が依頼の正本である
    # **満たされた**(2026-10-27 の便)—— 下限の層は 512層・210頁、全部 用紙 と 色調補正
    "req3_layer_kind": ("", ("source", "req3_body_blocks")),
    "added33_structure": ("`canvas_unit` と `v_canvas_px_w/h`"
                          "(行ごとの単位が決められない。T126)",
                          ("source", "canvas_px")),
    "db_all_layers": ("", None),
    "grad_v1_v2": ("", None),
    "delivery_png": ("", None),
    "body_initcolor": ("", None),
}


def audit_inline_needs(outdir: str) -> int:
    """**`populations.csv` に直に書いた行の `needs` も見る。**

    `sources.audit_needs()` は宣言した行(`SUPPLIER_REPORTED` / `ABSENT`)しか見ない。
    **`write_populations` が直に書く行は、その外に在った** ——
    2026-10-26 に `req3_layer_kind` の `needs` が
    「異なり数は `file_hashes` が無いので測れない」と残ったまま、
    **その sha256 は既に届いていた。**
    **点検の覆いが、守ると宣言した範囲より狭かった**(T134 (d) の再来)。

    **届かないもの**: `INLINE_NEEDS` に宣言していない行 /
    **材料が届いてもその問いに答えていない場合**(鍵が在ることと答えが在ることは別)。
    """
    have = {x["key"] for x in sources.LOCAL_CSV}
    with open(os.path.join(outdir, "populations.csv"), encoding="utf-8") as fh:
        rows = {r["population"]: r for r in csv.DictReader(fh)}
    bad = []
    print("  [`populations.csv` に直書きした行の `needs` の点検]")
    for key, (needs, cond) in sorted(INLINE_NEEDS.items()):
        r = rows.get(key)
        if r is None:
            bad.append((key, "行が無い", needs))
            continue
        written = r["needs"].strip()
        if cond is None:
            state = "満たされ方を宣言していない(取り直しの依頼など)"
            ok = True
        elif cond[0] == "column":
            filled = bool(r[cond[1]].strip())
            state = f"{cond[1]} は{'埋まっている' if filled else '空'}"
            ok = (not written) if filled else bool(written)
        else:
            arrived = cond[1] in have
            # **「未着」と「届いているが未統合」を言い分ける**(`sources` と同じ語で)
            pending = any(a.get("supplies") == cond[1]
                          for a in sources.ARRIVED_NOT_INTEGRATED)
            state = (f"材料 {cond[1]} は在る" if arrived else
                     f"材料 {cond[1]} は**届いているが未統合**" if pending else
                     f"材料 {cond[1]} は未着")
            ok = (not written) if arrived else bool(written)
        if not ok:
            bad.append((key, state, written))
        print(f"    {key:20s} {state:40s} needs={'空' if not written else '有り'}"
              f"  {'' if ok else '**食い違い**'}")
    for key, state, written in bad:
        print(f"    **{key}: {state} なのに needs が"
              f"{'残っている' if written else '空'}**")
    return 1 if bad else 0


def emit_occluders(outdir: str, root: str, drops):
    """遮蔽物の層。**層名は出さない。種別ごとの件数だけ。**

    **この表は全数の傾向を答えない** —— 層名が手元に在るのは納品の manifest が
    覆う頁だけ。**依頼6・7が来てから全数で引き直す。**
    """
    pages = occluders.load_named_layers(root, drops)
    rows, n_pages, n_layers, n_named = occluders.classify(pages, drops)
    # **両面の裏付け**(作者の指示 2026-10-27: 片面だけで済ませない)
    two = {r[0]: r for r in occluders.two_sided(pages, drops)}
    out = []
    for r in rows:
        t2 = two.get(r[0], [None, None, 0, 0, 0, ""])
        out.append(r + [t2[3], t2[4], t2[5],
                        f"{n_pages} 頁 / {n_layers} 層(名前が在る層 {n_named})",
                        "census track(納品の manifest から数えた)", bc_today()])
    written = write_csv(os.path.join(outdir, "occluder_classes.csv"),
                        ["occluder_class", "label", "found_by", "n_layers", "n_pages",
                         "n_signatures", "n_signature_only", "corroboration",
                         "population_note", "measured_by", "as_of"], out)
    print(f"  [遮蔽物] 層名が在る頁 {n_pages} / 層 {n_layers} / 名前が在る層 {n_named}")
    print("    **全数の傾向はこの表では出ない** —— 依頼6・7が来てから引き直す")
    print("    [両面の裏付け] 名前側 / 素性だけの層 / 判定")
    for r in out:
        print(f"      {r[0]:22s} {r[3]:4d} / {r[6]:4d}  {r[7]}")
    print("    **選んだ素性は何も特徴づけていない** —— "
          "統計量側には画素の分布の列が要る(未測定)")
    return written


def emit_balloon_two_sided(outdir: str, root: str, drops):
    """**フキダシの両面。** 作者の指示(2026-10-27)を、フキダシに当てて書き直した表。

    > **名前で見つけて統計量で裏付ける・統計量で見つけて名前で裏付ける。**
    > **どちらの順でもいいが、片面だけで済ませない方がよい。**

    **上の `occluder_classes.csv` の素性は失敗した**(`layer_type`/`composite`/`clip`/
    `has_mask` は何も特徴づけない)。**その失敗は残したまま、別の側をここに置く** ——
    **画素の分布から、節ごとに「その層が物として何であるか」を書いて導いた素性。**

    **母集団が2つ在る。混ぜない**:
    - **名前側は全数**(437,561層。依頼7)→ `occluder_vocab_full.csv`
    - **素性側は8頁**(画素の分布の列が在る範囲)→ この表
    **だから「両面一致」と言える範囲は8頁である。**
    """
    pages = occluders.load_named_layers(root, drops)
    rows = occluders.mechanism_two_sided(pages, drops)
    n_pages = len(pages)
    n_layers = sum(len(v) for v in pages.values())
    n_feat = sum(1 for ls in pages.values() for L in ls
                 if L.get("colours_q16") is not None)
    n_feat_pages = len({pg for pg, ls in pages.items() for L in ls
                        if L.get("colours_q16") is not None})
    note = (f"素性の列(`colours_q16`/`top_colour_share`/`sat_mean`)が在るのは "
            f"{n_feat_pages} 頁 / {n_feat} 層。層名が在るのは {n_pages} 頁 / "
            f"{n_layers} 層。**全数ではない**")
    out = [r + [note, "census track(納品の manifest の画素の分布から数えた)", bc_today()]
           for r in rows]
    written = write_csv(
        os.path.join(outdir, "occluder_balloon_two_sided.csv"),
        ["occluder_class", "label", "probe", "probe_what", "n_layers", "n_pages",
         "n_both_sides", "n_feature_only", "n_name_only", "reading",
         "population_note", "measured_by", "as_of"], out)
    print(f"  [フキダシの両面] 素性の列が在る {n_feat_pages} 頁 / {n_feat} 層")
    for r in out:
        print(f"      {r[0]:8s} {r[2]:18s} n={r[4]:>5} 両面={str(r[6]):>4} "
              f"素性だけ={str(r[7]):>4} 名前だけ={str(r[8]):>3} {r[9]}")
    print("    **両面一致と言えるのはこの頁の範囲だけ。** "
          "名前側の全数は `occluder_vocab_full.csv`")
    return written


def emit_vocab_full(outdir: str, path: str, drops):
    """**全数の名前側**(依頼7、437,561層 / 16,241 の異なる名前)。

    **層名は出さない** —— 出すのは**語彙に当たった名前の異なり数と、層・フォルダーの件数。**
    **`top_parent_name`(最頻の親の名前)は作者の作業名なので出さない。**
    異なり数 `n_parent_names` だけを出す。

    **頁の和を書かない** —— 同じ頁に2つの名前が在りうるので、名前ごとの `n_pages` の
    和は頁数ではない。**最大の1名の頁数だけを書く。**
    """
    rows = occluders.load_vocab_table(path, drops)
    if not rows:
        print("  [全数の名前側] **依頼7の表が無い。** この表は書かない")
        return 0
    # **年の札は白名簿に入れる前に形を強制する**(入れた語は点検から外れる)
    years = occluders.vocab_year_labels(rows)
    ALLOWED_PUBLIC_TOKENS.update(years)
    out = occluders.vocab_full(rows, drops)
    n_names = len(rows)
    note = (f"全 {fp.N_LAYERS:,} 層 / 名前の異なり {n_names:,}(依頼7)。"
            "**フォルダーの行を含む。** 素性側の裏付けは無い"
            "(全数には画素の分布の列が無い)")
    final = []
    for r in out:
        final.append([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8],
                      r[9], r[10] if len(r) > 10 else "", note,
                      "抽出側(um350)が集計 / 語彙は census track が書いた", bc_today()])
    written = write_csv(
        os.path.join(outdir, "occluder_vocab_full.csv"),
        ["occluder_class", "label", "n_names", "n_layers", "n_folders",
         "n_pages_top_name", "first_year_label", "last_year_label",
         "n_parent_names", "reading", "kinds_of_layers",
         "population_note", "measured_by", "as_of"], final)
    print(f"  [全数の名前側] 名前の異なり {n_names:,} / 年の札 {len(years)} を白名簿へ")
    for r in final:
        print(f"      {r[0]:24s} 名前 {str(r[2]):>4} / 層 {str(r[3]):>6} / "
              f"フォルダー {str(r[4]):>4}  {r[6]}→{r[7]}")
    print("    **名前で集めると年によって見つかる物が違う** —— "
          "一様な取りこぼしではなく**世代と相関した**取りこぼし")
    return written


def emit_front_position(outdir: str, path: str, drops):
    """**全数の構造の側**(依頼8、437,561層)。**遮蔽物は順序の性質である。**

    **名前で集めた層が、本当に手前に居るか。** 棒の前に**帰無値**を出す(規律9)——
    「手前に寄っている」は、**同じ群の層を1つ選んだときの前方割合**と比べて初めて言える。

    **`scope` が2つ在るのは、2つが別の量だから**(T138)——
    **フキダシは頁の中では手前寄りで、親の中では後ろ寄り**(文字がその上に乗る)。
    **同じ列名に2つの量を入れない。**
    """
    pages, aux, n_rows = occluders.load_structure(path, drops)
    if not pages:
        print("  [全数の重なり順] **依頼8 の構造の表が無い。** この表は書かない")
        return 0
    rows = occluders.front_position(pages, aux, drops)
    note = (f"全 {n_rows:,} 行のうち、**フォルダー以外で `z_global` を持つ層。** "
            f"`z_global` が空で重なり順に入らない非フォルダー層 {aux['n_no_z']} は除いた"
            "(供給元の言明: 木から辿れない層とその中の層)。"
            "**両形式を含む**(`.lip` を母集団に入れた。作者の決定 2026-10-27)—— "
            "**形式ごとの内訳は `n_layers_clip` / `n_layers_lip` の列に在る。**"
            "**`.lip` が在る年の札は限られている**ので"
            "(どの札かは `population_by_format.csv`)、"
            "**形式の内訳は世代の内訳でもある** —— この表では分離できない")
    out = [r + [note, "census track(供給元の `z_global` から数えた。"
                      "**向きは供給元の v2 の訂正に依る**)", bc_today()] for r in rows]
    written = write_csv(
        os.path.join(outdir, "occluder_front_position.csv"),
        ["occluder_class", "label", "matched_on", "scope", "n_layers", "n_pages",
         "n_layers_clip", "n_layers_lip",
         "median_front_share", "share_front_decile", "share_opacity_full",
         "share_visible_bit", "share_composite_normal", "share_ancestor_hidden",
         "reading", "population_note", "measured_by", "as_of"], out)
    print(f"  [全数の重なり順] {n_rows:,} 行 / 頁 {len(pages):,}"
          f" / 重なり順に入らない層 {aux['n_no_z']}")
    for r in out:
        print(f"      {r[0]:22s} {r[2]:10s} {r[3]:6s} n={r[4]:>6} "
              f"(clip {r[6]:>6} / lip {r[7]:>6}) "
              f"中央={r[8]} 前方1割={r[9]} 祖先が非表示={r[13]}  {r[14]}")
    print("    **手前に在ることは、重なっていることではない** —— "
          "何を隠しているかは画素が要る(保留中)")
    print(f"    **祖先のフォルダーが非表示の層 {aux['n_hidden_ancestor']:,}**(T143)"
          " —— **層自身が可視でも、描かれない。** "
          "**可視の列は層自身の値である**")
    return written


def emit_front_decile_kinds(outdir: str, path: str, drops):
    """**軸の端を占めているのは何か。** 但し書きの文を、数に替える表。

    **対照が保証するのは、比べた点の軸だけである**(基盤、2026-10-27)——
    線画は**中央では帰無値どおり(0.508)**だが**前方1割では帰無値を下回る(0.048)。**
    **だから前方1割に何が居るかを数える。**
    """
    pages, aux, n_rows = occluders.load_structure(path, drops)
    if not pages:
        return 0
    rows, n_dec, n_all, comb = occluders.front_decile_kinds(pages, aux, drops)
    note = (f"頁の中の前方割合。{occluders.FRONT_DECILE_RULE}。"
            f"前方1割 {n_dec:,} 層 / 群に入った層 {n_all:,}"
            f"({n_dec / n_all:.4f})。"
            f"**手で描いた線ではない2種**({'・'.join(comb['kinds'])})は"
            f"前方1割の {comb['share_of_decile']:.4f} を占め、"
            f"母集団では {comb['share_of_all']:.4f}"
            f"({comb['n_in_all']:,} 層)—— **この2つは数から出した割合で、"
            f"印字された列を足したものではない**")
    out = [r + [note, "census track(依頼8 の `z_global` から数えた)", bc_today()]
           for r in rows]
    written = write_csv(
        os.path.join(outdir, "front_decile_kinds.csv"),
        ["v_layer_kind", "n_layers", "n_layers_clip", "n_layers_lip",
         "share_of_decile", "share_of_kind",
         "share_of_all", "reading", "population_note", "measured_by", "as_of"], out)
    print(f"  [前方1割を占めている種類] {occluders.FRONT_DECILE_RULE}")
    for r in out:
        print(f"      {r[0]:16s} {r[1]:>7} (clip {r[2]:>6} / lip {r[3]:>6}) "
              f"前方1割の {r[4]} / その種類の {r[5]}"
              f" / 全体では {r[6]}  {r[7]}")
    print(f"      束ねた2種 {'・'.join(comb['kinds'])}: 前方1割の "
          f"{comb['share_of_decile']:.4f} / 母集団の {comb['share_of_all']:.4f}"
          f"({comb['n_in_all']:,} 層)")
    print("    **境界の値は列名が運ばない** —— "
          "基盤が `<= 0.1` で測って 529層ずれた。**母集団の欄に書いてある**")
    print("    **束ねた割合は数から出す** —— "
          "印字された小数3桁を足すと 0.086 になるが、数から出すと 0.0853")
    return written


def emit_population_by_format(outdir: str, path: str, n1_path: str, drops):
    """**母集団を年 × 形式で出す表。** 作者の決定(2026-10-27): **`.lip` を入れる。**

    **この表が在るまで、`.lip` は入っていながら書かれていなかった** ——
    依頼7と依頼8 の「全数 437,561層」は **`.clip` 312,898 + `.lip` 124,663** の和で、
    **公開した表のどこにも形式の列が無かった。**
    **決定が変えたのは材料ではなく、書いてあることである**(T139)。

    **頁の数と層の数は別の便から来る**(頁は便 n1、層とフォルダーは依頼8)——
    依頼8 には**層の在る頁**しか無いので、**読めなかった4頁はそこに出てこない。**
    だから `n_pages` と `n_pages_with_layers` を**別の列**にしてある(T138)。
    """
    pages, aux, n_rows = occluders.load_structure(path, drops)
    if not pages:
        print("  [母集団の形式ごとの内訳] **依頼8 の構造の表が無い。** この表は書かない")
        return 0
    n1 = occluders.load_pages_by_year(n1_path, drops)
    if not n1:
        print("  [母集団の形式ごとの内訳] **便 n1 が無い。** この表は書かない")
        return 0
    # **年の札は白名簿に入れる前に形を強制する**(`load_pages_by_year` と `path_year`
    # の両方で `^\d{4}[a-z]?$` を強制してある)
    ALLOWED_PUBLIC_TOKENS.update({yr for yr, _ in n1})
    ALLOWED_PUBLIC_TOKENS.update({yr for yr, _ in aux["fmt_year"]})
    rows = occluders.population_by_format(aux, n1, drops)
    note = (f"頁の数は便 n1、層とフォルダーの数は依頼8 の表({n_rows:,} 行)。"
            "**どちらも同じ DB の同じ走行**なので、**一致は一致であって"
            "独立な確認ではない**(T132)。"
            "**`.lip` の画素は1頁も無い** —— 便 n4 の `rendered` は全行が空で、"
            "抽出側の言明は**「描けなかったのではなく、描いていない」。**"
            "だから**素性の側(画素の分布)の `.lip` は 0 頁**で、"
            "**0 と測ったのではなく材料が無い**(T126)。"
            "**`sum_n_layers_incl_folders` はフォルダーの行を含む** —— "
            "**同じ割合を非フォルダーの分母で出すと別の数になる**"
            "(2026-10-27、基盤と 66.3% 対 70.0% でずれた。**どちらも正しく、"
            "どちらも「層」と書いていた**)")
    out = [r + [note, "抽出側(um350)が出した2便 / census track が年 × 形式に組んだ",
                bc_today()] for r in rows]
    written = write_csv(
        os.path.join(outdir, "population_by_format.csv"),
        ["grain", "year_label", "format", "n_pages", "n_read_error",
         "n_pages_with_layers", "sum_n_layers_incl_folders", "sum_n_folders",
         "reading", "population_note", "measured_by", "as_of"], out)
    print(f"  [母集団の形式ごとの内訳] 年 × 形式 {sum(1 for r in rows if r[0] == 'year')} 行"
          f" / 拡張子が2形式でない行 {aux['n_bad_fmt']}"
          f" / 年の札の形でない行 {aux['n_bad_year']}")
    for r in out:
        if r[0] == "format_total":
            print(f"      {r[2]:5s} 頁 {r[3]:>6}(層が在る {r[5]:>6})"
                  f" / 層 {r[6]:>7} / フォルダー {r[7]:>7}  {r[8]}")
    # **3便の突き合わせ。** 合わないまま公開しない
    bad = 0
    print("    [便どうしの突き合わせ]")
    for what, mine, theirs in occluders.check_formats(
            aux, occluders.N2_DEFAULT, occluders.N4_DEFAULT, drops):
        ok = mine == theirs
        bad += 0 if ok else 1
        print(f"      {what}: こちら {mine:,} / 便 {theirs:,}  "
              f"{'一致' if ok else '**食い違い**'}")
    if bad:
        print(f"      **{bad} 件 食い違っている。** 読み取りか便のどちらかが違う")
    print("    **`.lip` が在る年の札は限られている** —— "
          "**形式ごとの差を世代の差と読み分けられない**(この表では分離できない)")
    return written


def emit_folder_contents(outdir: str, path: str, drops):
    """**名前がフォルダーに付いている類**の中身(依頼8)。

    **`黒消し` は層では0件、フォルダーで 609。** 遮蔽しているのは**その中の層**で、
    **フォルダーの数を遮蔽物の数と呼べない。** 中が空のフォルダーも数える(規律9)。
    """
    pages, aux, n_rows = occluders.load_structure(path, drops)
    if not pages:
        return 0
    rows = occluders.folder_contents(aux)
    out = [r + [f"全 {n_rows:,} 行(フォルダーの行を含む)",
                "census track(依頼8 の `parent_id` から数えた)", bc_today()]
           for r in rows]
    written = write_csv(
        os.path.join(outdir, "occluder_folder_contents.csv"),
        ["occluder_class", "label", "n_folders", "n_folders_clip", "n_folders_lip",
         "n_folders_with_layers",
         "n_folders_empty", "n_folders_hidden", "n_child_layers", "n_pages",
         "author_reading", "reading",
         "population_note", "measured_by", "as_of"], out)
    print("  [フォルダーに付いた名前の中身]")
    for r in out:
        print(f"      {r[0]:22s} フォルダー {r[2]:>4}"
              f"(clip {r[3]:>4} / lip {r[4]:>4}) / 中に層が在る {r[5]:>4}"
              f" / 空 {r[6]:>4} / 直下の層 {r[7]:>5}")
        if r[9]:
            print(f"        {r[9]}")
    return written


def bc_today() -> str:
    return today()


def check_full_population_totals() -> int:
    """供給元の2つの表が、どちらも宣言した全数に合うか。

    **届く**: こちらの転記の誤り / 供給元の表の足し算の誤り。
    **届かない**: DB そのものの誤り(こちらに DB は無い)。
    """
    a = sum(v["n_layers"] for v in fp.VALUES)
    b = sum(m["n_layers"] for m in fp.MASK_BY_KIND)
    print(f"  [全数の表の突き合わせ] 宣言 {fp.N_LAYERS:,} 層")
    print(f"    layer_type の値 12行の合計 {a:,}  "
          f"{'一致' if a == fp.N_LAYERS else '**食い違い**'}")
    print(f"    マスクの種類 5行の合計   {b:,}  "
          f"{'一致' if b == fp.N_LAYERS else '**食い違い**'}")
    print("    **これはこちらの転記と供給元の足し算の点検で、DB の点検ではない**")
    return 0 if (a == fp.N_LAYERS and b == fp.N_LAYERS) else 1


# ---------------------------------------------------------------- 集計(1回だけ)

def aggregate(req3_rows, req3_pages, a33_rows):
    """**公開と全量が同じここから出る。** 鍵は (公開次元, 全量次元) の組。"""
    layer_cells = collections.defaultdict(collections.Counter)
    page_cells = collections.defaultdict(lambda: collections.Counter())

    for r in req3_rows:
        pub = ("req3_layer_kind", r["year_label"], r["v_layer_kind"], r["layer_type"])
        pri = (r["work"], r["subdir"], r["page_path"])
        c = layer_cells[(pub, pri)]
        c["n_layers"] += 1
        c["n_rendered" if r["rendered"] == "true" else "n_not_rendered"] += 1
        c["n_drawn_" + r["drawn_bucket"]] += 1
        if r["has_image_material"] == "true":
            c["n_has_image_material"] += 1

    for page_path, p in req3_pages.items():
        pub = ("req3_layer_kind", p["year_label"], "", "")
        pri = (p["work"], p["subdir"], page_path)
        c = page_cells[(pub, pri)]
        c["n_pages"] += 1
        c["n_pages_without_work"] += 0 if p["has_work"] else 1
        c["ext_" + p["ext"].lstrip(".")] += 1

    for r in a33_rows:
        pub = ("added33_structure", r["year_label"], "", "")
        pri = (r["work"], r["subdir"], r["page_path"])
        c = page_cells[(pub, pri)]
        c["n_pages"] += 1
        c["n_pages_without_work"] += 0 if r["has_work"] else 1
        c["ext_" + r["ext"].lstrip(".")] += 1
        c["sum_n_layers_incl_folders"] += r["n_layers"]
        c["sum_n_folders"] += r["n_folders"]
        c["sum_n_text_layers"] += r["n_text_layers"]
        c["sum_n_image_material_layers"] += r["n_image_material_layers"]

    return layer_cells, page_cells


def project(cells, keep_private: bool):
    """公開側は **PUBLIC_DIMS だけを持つ鍵**へ畳む。全量側は両方を持つ。

    公開側の行 dict は **名前の欄を一度も持たない** —— 落とす処理が無い。
    """
    out = collections.defaultdict(collections.Counter)
    for (pub, pri), counter in cells.items():
        key = pub + pri if keep_private else pub
        out[key].update(counter)
    dims = PUBLIC_DIMS + PRIVATE_DIMS if keep_private else PUBLIC_DIMS
    return dims, out


def write_csv(path: str, header, rows) -> int:
    """**ヘッダーと行の幅が合わないまま書かせない。**

    2026-10-27 に3度目を踏んだ —— `str.replace` は**既定で全件置換**なので、
    同じ並びの列名を別の表でも使っていると、**意図した表の外側も書き換わる。**
    (`METRIC_COLS` を直したら遮蔽物の表のヘッダーにも入り、**14 列のヘッダーに
    11 列の行**という形になった。)点検は走行の最後に鳴るが、これは**書く所で落ちる。**
    """
    header = list(header)
    for i, row in enumerate(rows):
        if len(row) != len(header):
            raise SystemExit(
                f"{path}: **ヘッダー {len(header)} 列 対 行 {len(row)} 列**"
                f"(行 {i + 2})\n  ヘッダー: {header}\n  行: {list(row)[:6]} ...\n"
                "  **列を足したら、その表のヘッダーと行の両方を直すこと。**")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        n = 0
        for row in rows:
            w.writerow(row)
            n += 1
    return n


def emit_tables(outdir: str, layer_cells, page_cells, keep_private: bool, n_works_by):
    written = {}
    ldims, lagg = project(layer_cells, keep_private)
    pdims, pagg = project(page_cells, keep_private)

    metrics_l = ["n_layers", "n_rendered", "n_not_rendered",
                 "n_drawn_gt0", "n_drawn_zero", "n_drawn_blank",
                 "n_has_image_material"]
    drawn_cols = ("n_drawn_gt0", "n_drawn_zero", "n_drawn_blank")
    rows = []
    suppressed_layers = collections.Counter()
    for key in sorted(lagg):
        c = lagg[key]
        kind = key[ldims.index("v_layer_kind")]
        reason = SUPPRESSED.get(("drawn_ratio", kind))
        usable, why = DRAWN_RATIO_USABLE.get(kind, DRAWN_RATIO_USABLE_DEFAULT)
        vals = []
        for m in metrics_l:
            if reason and m in drawn_cols:
                vals.append("")          # **空欄。0 ではない** —— 数えていない
            else:
                vals.append(c[m])
        if reason:
            suppressed_layers[f"drawn_ratio / {kind}: {reason}"] += c["n_layers"]
        rows.append(list(key)
                    + [calendar_year(key[ldims.index("year_label")])]
                    + vals + [usable, why])
    written["layers_by_year_kind_type.csv"] = write_csv(
        os.path.join(outdir, "layers_by_year_kind_type.csv"),
        list(ldims) + ["calendar_year"] + metrics_l
        + ["drawn_ratio_usable", "drawn_ratio_usable_why"], rows)
    print("  [集計しなかった列(直る予定があるため)]")
    if suppressed_layers:
        for reason, n in suppressed_layers.most_common():
            print(f"    {n:7d} 層  {reason}")
    else:
        print("    なし(0件)")
    for (col, scope), reason in sorted(SUPPRESSED.items()):
        if scope == "*":
            print(f"    (材料に無い)  {col} / 全種別: {reason}")

    metrics_p = ["n_pages", "n_pages_without_work", "ext_clip", "ext_lip",
                 "sum_n_layers_incl_folders", "sum_n_folders",
                 "sum_n_text_layers", "sum_n_image_material_layers"]
    # 頁の表は種別で分かれない。**空の次元列を公開物に出さない**(読み手が意味を探す)
    page_dims = [d for d in pdims if d not in ("v_layer_kind", "layer_type")]
    page_idx = [pdims.index(d) for d in page_dims]
    rows = []
    not_measured = collections.Counter()
    for key in sorted(pagg):
        c = pagg[key]
        pop = key[0]
        supplied = POPULATION_METRICS[pop]
        row = [key[i] for i in page_idx]
        row.append(calendar_year(key[1]))
        for m in metrics_p:
            if m in supplied:
                row.append(c[m])
            else:
                row.append("")              # **測っていない。0 ではない**
                not_measured[f"{pop}: {m}"] += 1
        if not keep_private:
            row.append(n_works_by[(pop, key[1])]["n_works"]
                       if "n_works" in supplied else "")
        rows.append(row)
    header = (page_dims + ["calendar_year"] + metrics_p
              + ([] if keep_private else ["n_works"]))
    print("  [その母集団では測っていない量(**空欄で出す。0 にしない**)]")
    if not_measured:
        for what, n in sorted(not_measured.items()):
            print(f"    {n:7d} 行  {what}")
    else:
        print("    なし(0件)")
    written["pages_by_year.csv"] = write_csv(
        os.path.join(outdir, "pages_by_year.csv"), header, rows)
    return written


# ------------------------------------------------- 公開物の点検(印字が無い走行は commit しない)

def private_vocabulary(req3_pages, a33_rows):
    """**公開してはいけない語**を材料そのものから組む。

    年の区画名は公開するので除く(段 A は年ごとの件数を許す)。
    戻り値は (長い語, 短い語) —— **短い語は点検の盲点**として別に返す:
    `1` `2` `a` `x` `work` `tmp` という名のフォルダーが実在するので、
    **2文字以下の名前は件数や生の値と見分けが付かない。** 盲点は隠さずに数えて印字する。
    """
    vocab = set()
    for p in list(req3_pages.values()) + list(a33_rows):
        year = p["year_label"]
        ALLOWED_PUBLIC_TOKENS.add(year)
        for seg in p["page_path"].split("\\"):
            if seg and seg.lower() not in ("f:", "manuscripts") and seg != year:
                vocab.add(seg)
                vocab.add(os.path.splitext(seg)[0])
    vocab.discard("")
    vocab |= set(AUTHOR_LAYER_NAMES)
    vocab -= ALLOWED_PUBLIC_TOKENS
    allowed = set(PROSE_EXCEPTIONS) | set(SYSTEM_LAYER_NAMES)
    long_words = ({v for v in vocab if len(v) >= 3} | set(AUTHOR_LAYER_NAMES)) - allowed
    short_words = vocab - long_words - allowed
    return long_words, short_words


class ProseMatcher:
    """散文に固有名が出ているか。**640語を1本の正規表現に束ねる。**

    1語ずつ `re.search` を回すと公開物 1枚の点検で 36.7 秒かかり、対照5腕で3分を超えた
    (2026-10-26 に実測)。**遅すぎて回されない点検は、回されない点検**なので束ねる。

    **ASCII の名前は語境界を要求する** —— `add` という名のフォルダーが実在し、
    素の部分一致では自分で書いた「added33」に当たって全行が鳴る(同日に実際そうなった)。
    日本語の名前には語境界が無いので素の部分一致で見る。

    **届かないもの**: ASCII の名前が英数字や `_` に直接挟まれている形
    (`x_<名前>_y`)。語境界が立たないので見えない。
    """

    def __init__(self, words):
        ascii_words = sorted((w for w in words if w.isascii()), key=len, reverse=True)
        self.other = [w for w in words if not w.isascii()]
        self.pat = None
        if ascii_words:
            self.pat = re.compile(r"(?<![0-9A-Za-z_])(?:"
                                  + "|".join(re.escape(w) for w in ascii_words)
                                  + r")(?![0-9A-Za-z_])")

    def find(self, text: str):
        """当たった語を返す。当たらなければ None。"""
        if self.pat is not None:
            m = self.pat.search(text)
            if m:
                return m.group(0)
        for w in self.other:
            if w in text:
                return w
        return None


# 公開 CSV の列の種類。**どの列も、ここに宣言が無ければ指摘になる**
# (宣言の無い列を黙って通すと、新しい列を足した日に点検が静かに緩む)
METRIC_COLS = ("n_both_sides", "n_feature_only", "n_name_only",
               "n_names", "n_folders", "n_pages_top_name", "n_parent_names",
               "n_folders_with_layers", "n_folders_empty", "n_folders_hidden",
               "n_child_layers",
               "n_pages", "n_distinct_pages", "n_layers", "n_rendered", "n_not_rendered",
               # **形式ごとの内訳**(`.lip` を母集団に入れた。作者の決定 2026-10-27)
               "n_layers_clip", "n_layers_lip",
               "n_folders_clip", "n_folders_lip",
               "n_read_error", "n_pages_with_layers",
               "n_drawn_gt0", "n_drawn_zero", "n_drawn_blank", "n_has_image_material",
               "n_pages_without_work", "n_works", "n_rows",
               "ext_clip", "ext_lip",
               "sum_n_layers_incl_folders", "sum_n_folders",
               "sum_n_text_layers", "sum_n_image_material_layers",
               "bit", "layer_type_and_2",
               "n_png_files", "n_distinct_page_keys", "n_png_names",
               "n_signatures", "n_signature_only",
               "n_layer_type_bit1", "n_layer_type_eq2", "n_layer_type_other",
               "n_png_a", "n_png_b", "n_distinct_pixels_a", "n_distinct_pixels_b",
               "n_common_names", "n_common_names_bytes_differ",
               "n_pixels_in_both", "n_pixels_in_both_same_name",
               "n_pixels_in_both_renamed", "n_pixels_only_a", "n_pixels_only_b",
               "n", "cluster_size", "n_clusters")
# 形式の列(白名簿は下の `allow` で宣言する)
# **件数ではなく実測値の列。** 小数でよい(件数が小数なら誤りだが、画布の寸法は小数が正しい)
VALUE_COLS = ("canvas_w", "canvas_h", "dpi",
              "share_of_decile", "share_of_kind", "share_of_all",
              # **割合の列。** 件数ではないので小数が正しい
              "median_front_share", "share_front_decile", "share_opacity_full",
              "share_visible_bit", "share_composite_normal",
              "share_ancestor_hidden")
# 値の集合が宣言できる列(**白名簿。** 他の値が出たら指摘)
DIM_COLS = ("population", "year_label", "v_layer_kind", "layer_type",
            "drawn_ratio_usable", "volatile",
            "is_folder", "has_frame_line", "has_mask", "has_stored_blocks",
            "alpha_max_zero", "rc_bucket", "init", "kind", "bits", "confidence",
            "delivery_group", "delivery_group_a", "delivery_group_b", "item",
            "is_mask", "content_match", "canvas_unit_resolved", "quantity", "format",
            "bound_class", "calendar_year", "occluder_class", "found_by",
            "is_drawing_layer", "gap_bucket",
            "probe", "first_year_label", "last_year_label",
            # **`scope` と `grain` は別の列である** —— 同じ列名に2つの量を入れない(T138)。
            # `scope` は前方割合を測った群(頁 / 親)、`grain` は母集団の表の行の粒度
            "matched_on", "scope", "grain")
# 手で書いた散文。**ここが実際に名前が漏れる場所**なので部分一致で見る
PROSE_COLS = ("predicate", "caveat", "needs", "note", "answers", "not_answers",
              "detail", "drawn_ratio_usable_why", "column", "trap", "by", "as_of",
              "measured_by", "source", "supplier", "sha256",
              "meaning", "evidence", "content", "direction", "material_kind", "why",
              "label", "what", "not_what", "share", "category", "rule",
              "probe_what",
              "population_note", "corroboration",
              # **作者の判断を測定の列に混ぜない。** 別の列にして、散文として点検する
              "author_reading",
              "reading", "kinds_of_layers")


def check_column_classes() -> None:
    """**1つの列名が2つの種類に宣言されていないか。走行の最初に落ちる。**

    **同じ誤りを2回したなら、それは器具の形である**(2026-10-26)——
    `kind` を「出所の種類」と「層の種類」に、`content` を「same/different」と
    「中身の説明」に使い、**2回とも点検の最後で鳴った。**
    どちらも直し方は同じ(片方を改名する)。

    **列の種類ごとの点検は、1つの列名が1つの量を指すことに依っている。**
    2つを指すと、片方の規則がもう片方に当たる。だから**宣言の段で落とす。**
    """
    classes = {"件数": METRIC_COLS, "実測値": VALUE_COLS,
               "次元": DIM_COLS, "散文": PROSE_COLS}
    seen = {}
    clash, dup = [], []
    for label, cols in classes.items():
        for c in cols:
            if c in seen:
                (dup if seen[c] == label else clash).append((c, seen[c], label))
            seen[c] = label
    # **同じ種類に2回は衝突ではない**(無害な重複)。言い分けないと、
    # 点検が自分の指摘の意味を取り違える(2026-10-26 に実際そうなった)
    if dup:
        print("  [列の宣言が重複] " + " / ".join(f"{c!r}(「{a}」に2回)"
                                                 for c, a, _ in dup))
    if clash:
        lines = "\n".join(f"  列 {c!r} が「{a}」と「{b}」の両方に宣言されている"
                           for c, a, b in clash)
        raise SystemExit(
            f"**列名が2つの種類に宣言されている**\n{lines}\n"
            "  **片方を改名すること。** 1つの列名は1つの量を指す")


def audit_public(outdir: str, vocab, verbose: bool = True):
    """`census/` の**全ファイルの全セル**を読み返し、固有名を含む数を印字する。

    **印字が無い走行の出力を commit しない**(`census/README.md` 規則4)。

    **この点検が届く範囲**(名前より狭く書く。T132):

    - 件数の列 —— 数字か空欄以外が入っていないこと。**名前は数字の列には入れられない**
    - 次元の列 —— 宣言した値の**白名簿**に入っていること。宣言外の値はすべて指摘
    - 散文の列 —— **3文字以上**の固有名を部分一致で探す
    - **届かないもの**: 2文字以下の名前(件数と見分けが付かない)、
      材料に出てこない名前、名前を含まない別名(別名はそもそも入れない規則)

    戻り値は (終了コード, 指摘の一覧)。**印字の文字列ではなく構造を返す** ——
    対照が printf を読むと、印字を20件で切った日に対照が黙る。
    """
    long_words, short_words = vocab
    matcher = ProseMatcher(long_words)
    allow = {
        "population": set(ALLOWED_POPULATIONS) | {"population"},
        "year_label": set(ALLOWED_PUBLIC_TOKENS) | {"year_label", ""},
        "calendar_year": {"calendar_year", ""},     # 残りは数字として見る
        "v_layer_kind": set(ALLOWED_LAYER_KINDS) | {"v_layer_kind", ""},
        "layer_type": {"layer_type", ""},          # 残りは数字として見る
        "drawn_ratio_usable": {"drawn_ratio_usable", "no", "yes", "unverified", ""},
        "volatile": {"volatile", "yes", "no", ""},
        "is_folder": {"is_folder", "yes", "no", ""},
        "has_frame_line": {"has_frame_line", "yes", "no", ""},
        "has_mask": {"has_mask", "yes", "no", ""},
        "has_stored_blocks": {"has_stored_blocks", "yes", "no", ""},
        "alpha_max_zero": {"alpha_max_zero", "yes", "no", ""},
        "rc_bucket": {"rc_bucket", "0.0", ">0", "manifest に無い", ""},
        "init": {"init", ""},                       # 残りは数字として見る
        "kind": {"kind", fp.FILL_LABEL, *fp.SET_LABELS.values(),
                 "(どの種類にも無し)"},
        "bits": {"bits", "—", ""},                  # `512+256+32` の形
        "confidence": {"confidence", "確か(全数で例外0)", "確か。**ただし1種類だけ外れる**",
                       "不明", ""},
        "delivery_group": {"delivery_group",
                           *{g for g, _ in delivery.GROUPS.values()}, ""},
        "delivery_group_a": {"delivery_group_a",
                             *{g for g, _ in delivery.GROUPS.values()}, ""},
        "delivery_group_b": {"delivery_group_b",
                             *{g for g, _ in delivery.GROUPS.values()}, ""},
        "item": {"item", *delivery.ITEMS.values(), ""},
        "is_mask": {"is_mask", "yes", "no", ""},
        "content_match": {"content_match", "same", "different", ""},
        "canvas_unit_resolved": {"canvas_unit_resolved", "no", "yes", ""},
        "quantity": {"quantity", "n_files", "n_distinct", "excess",
                     "files_in_clusters", "clusters", ""},
        "format": {"format", "clip", "lip", "both", ""},
        "bound_class": {"bound_class", *{c for c, _, _ in BOUND_CLASSES}, ""},
        "occluder_class": {"occluder_class", occluders.BASELINE_KEY,
                           *{k for k, _, _, _ in occluders.CLASSES},
                           *{k for k, _, _ in occluders.FRONT_CONTROLS}, ""},
        "matched_on": {"matched_on", "layer_name", "top_name", "other_names",
                       "ancestors", "baseline", ""},
        "scope": {"scope", *{k for k, _ in occluders.SCOPES}, ""},
        "grain": {"grain", "year", "format_total", ""},
        "found_by": {"found_by", "名前の規則(層名とフォルダーの鎖)", "種別", ""},
        "is_drawing_layer": {"is_drawing_layer", "yes", "no", ""},
        "gap_bucket": {"gap_bucket", *{b for _, b in faint.GAP_BUCKETS}, ""},
        "probe": occluders.probe_tokens(),
        # **年の札は `occluder_vocab_full.csv` を書くときに形を強制して入れる**
        "first_year_label": set(ALLOWED_PUBLIC_TOKENS) | {"first_year_label", ""},
        "last_year_label": set(ALLOWED_PUBLIC_TOKENS) | {"last_year_label", ""},
    }
    hits = []
    n_cells = n_files = 0
    for name in sorted(os.listdir(outdir)):
        if not name.endswith(".csv"):
            continue
        path = os.path.join(outdir, name)
        n_files += 1
        with open(path, encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            for lineno, row in enumerate(reader, start=2):
                for col, cell in enumerate(row):
                    n_cells += 1
                    s = cell.strip()
                    colname = header[col] if col < len(header) else f"<列{col}>"
                    if "\\" in s:
                        hits.append((name, lineno, colname,
                                     "パス区切り(`\\`)を含む", s[:70]))
                        continue
                    if s.lower().startswith("f:"):
                        hits.append((name, lineno, colname,
                                     "ドライブ名(`F:`)で始まる", s[:70]))
                        continue
                    if colname in METRIC_COLS:
                        if s and not s.lstrip("-").isdigit():
                            hits.append((name, lineno, colname,
                                         "件数の列に数字でない値", s[:70]))
                    elif colname in VALUE_COLS:
                        try:
                            if s:
                                float(s)
                        except ValueError:
                            hits.append((name, lineno, colname,
                                         "実測値の列に数でない値", s[:70]))
                    elif colname in DIM_COLS:
                        if (s and s not in allow[colname]
                                and not s.lstrip("-").isdigit()
                                and not (colname == "bits"
                                         and all(t.isdigit() for t in s.split("+")))):
                            hits.append((name, lineno, colname,
                                         "次元の列に宣言外の値(白名簿にない)", s[:70]))
                    elif colname in PROSE_COLS:
                        w = matcher.find(s)
                        if w is not None:
                            hits.append((name, lineno, colname,
                                         f"散文に固有名({len(w)}文字)", s[:70]))
                    else:
                        hits.append((name, lineno, colname,
                                     "**種類が宣言されていない列**", s[:70]))
    if verbose:
        print(f"\n[公開物の点検] {outdir}/")
        print(f"  禁止語彙 {len(long_words)} 語(3文字以上)"
              f" / 年の区画名 {len(ALLOWED_PUBLIC_TOKENS)} 語は公開してよい語として除外")
        print(f"  散文に出てよい語 {len(PROSE_EXCEPTIONS)} 語"
              f"(拡張子と英単語。**語ごとに理由を宣言済み**): "
              + " / ".join(sorted(PROSE_EXCEPTIONS)))
        print(f"  出してよい層名 {len(SYSTEM_LAYER_NAMES)} 語"
              f"(**アプリが付ける名前。作者の作業名は {len(AUTHOR_LAYER_NAMES)} 語を禁止**): "
              + " / ".join(sorted(SYSTEM_LAYER_NAMES)))
        print(f"  **この点検の盲点: 2文字以下の固有名 {len(short_words)} 語** ——"
              f" 件数や生の値と見分けが付かないので見ていない。"
              f"ASCII の名前は語境界を要求するので `x_名前_y` の形も見えない")
        print(f"  検査: {n_files} ファイル / {n_cells} セル"
              f" / 宣言した列の種類: 件数 {len(METRIC_COLS)}・実測値 {len(VALUE_COLS)}"
              f"・次元 {len(DIM_COLS)}・散文 {len(PROSE_COLS)}")
        print(f"  **指摘: {len(hits)}**")
        by_reason = collections.Counter(h[3] for h in hits)
        for reason, n in by_reason.most_common():
            print(f"    {n:6d}  {reason}")
        for h in hits[:10]:
            print(f"      {h[0]}:{h[1]} 列 {h[2]} —— {h[3]}: {h[4]}")
        if len(hits) > 10:
            print(f"      (ほか {len(hits) - 10} 件。**全件は戻り値に在る**)")
    return (1 if hits else 0), hits


# ------------------------------------------------------------------ 据え置きの表

def distinct_pages(pages, sha_by_path) -> tuple:
    """その母集団の**異なり頁数**。**ファイル数とは別の量**(T96 の形)。

    全 9,653 の sha256 が届いたので、**手元の母集団についても測れる** ——
    それまで `n_distinct_pages` は全行空欄で、`needs` に「測れない」と書いてあった。
    """
    have = [p for p in pages if p in sha_by_path]
    missing = len(pages) - len(have)
    return len(set(sha_by_path[p] for p in have)), missing


def write_populations(outdir: str, req3_rows, req3_pages, a33_rows, as_of: str,
                      sha_by_path=None):
    """**数の隣に母集団(述語)を書く。** 測った者の名も書く(T132)。"""
    header = ["population", "predicate", "n_pages", "n_distinct_pages", "n_layers",
              "measured_by", "as_of", "needs", "caveat"]
    rows = []
    for s in sources.SUPPLIER_REPORTED:
        rows.append([pop(s["key"]), s["predicate"],
                     s["n_pages"] if s["n_pages"] is not None else "",
                     "", s["n_layers"] if s["n_layers"] is not None else "",
                     s["measured_by"], s["as_of"], s["needs"], s["caveat"]])
    sha_by_path = sha_by_path or {}
    d_req3, miss_req3 = distinct_pages(list(req3_pages), sha_by_path)
    d_a33, miss_a33 = distinct_pages([r["page_path"] for r in a33_rows], sha_by_path)
    rows.append([pop("req3_layer_kind"),
                 "依頼3の872頁の**フォルダー以外の全層**",
                 len(req3_pages), d_req3 or "", len(req3_rows),
                 f"census track({as_of} にこの CSV を数えた"
                 f"。異なりは sha256 の列から)", as_of,
                 INLINE_NEEDS["req3_layer_kind"][0],
                 "**供給元の集計と一致しても独立な確認ではない**(同じ配列。T132)。"
                 f"**872頁のうち異なりは {d_req3}** —— "
                 f"**{len(req3_pages) - d_req3}頁は同じ中身の重複**で、"
                 "**層の数はファイル基準なのでその分膨らんでいる**"
                 + (f"。**sha256 が無い頁 {miss_req3}**" if miss_req3 else "")])
    rows.append([pop("db_all_layers"),
                 f"構造 DB の全数 —— {fp.N_PAGES} 頁 / {fp.N_LAYERS} 層"
                 "(`.clip` + `.lip`。値の種類と振る舞いは同じ)",
                 fp.N_PAGES, "", fp.N_LAYERS, fp.SUPPLIER, fp.AS_OF,
                 "",
                 "`layer_type` のビットと値の表は受領済み。"
                 "**ここでは測れない**(DB が手元に無い)。"
                 "合うのは**こちらの転記と供給元の足し算**だけ"])
    rows.append([pop("grad_v1_v2"),
                 "グラデの v1/v2 を**層を鍵に**突き合わせた 258層",
                 "", "", 258,
                 f"census track({as_of} に供給元の対応表から数え直した)", as_of,
                 "",
                 "対応表は受領済み。**名前を鍵にすると 152枚が「違う」と出る** —— "
                 "`_grad{N}` の `N` は頁の中の順番で、v2 で103層が加わって後ろにずれた"])
    rows.append([pop("delivery_png"),
                 "`data/incoming_manuscript/` に届いた PNG(**13の納品**)",
                 "", "", "",
                 f"census track({as_of} に納品の木を歩いた)", as_of,
                 "",
                 "`data/` は手元に在る。**単位は PNG の枚数で、層の数ではない。** "
                 "何が1枚かが納品ごとに違う(mask だけ / 本体だけ / 両方)。"
                 "**同名で中身が違う PNG が 256枚**在る"])
    rows.append([pop("body_initcolor"),
                 "トーン・グラデ・ベタの本体の `InitColor`(1層1行)",
                 "", "", 2095, "census track(2026-10-26 に行から数え直した)",
                 fp.AS_OF,
                 "",
                 "**要約の転記ではない** —— `raster_cover = 0.0` の 132層が"
                 "96(init 255)+ 36(init 0)に分かれることを行から数え直した"])
    for a in sources.ABSENT:
        rows.append([pop(a["key"]), a["predicate"], a["n_pages"], "", "",
                     a["measured_by"], a["as_of"], a["needs"], a["caveat"]])
    rows.append([pop("added33_structure"),
                 "後から入った33頁の構造。**層数はフォルダーを含む**",
                 len(a33_rows), d_a33 or "", "",
                 f"census track({as_of} にこの CSV を数えた)", as_of,
                 INLINE_NEEDS["added33_structure"][0],
                 "**依頼3の層数と比べない**(述語が違う)"])
    return write_csv(os.path.join(outdir, "populations.csv"), header, rows)


def write_provenance(outdir: str, as_of: str):
    header = ["source", "material_kind", "supplier", "as_of", "n_rows", "sha256",
              "volatile", "note"]
    rows = []
    ver = instrument_version()
    rows.append(["(この走行の器具)", "**器具の版**", "census track", as_of, "",
                 ver["commit"], "",
                 f"**この数を計算した器具の版。** 器具のファイルに未コミットの変更: "
                 f"{ver['dirty']}"
                 + (f"({ver['dirty_files']} ファイル)。"
                    "**版が器具を同定しない**" if ver["dirty"] == "yes" else "。"
                    "**版が器具を同定する**")
                 + " —— **報告でこの数を引くときは版も引く**(T139)"])
    for s in sources.LOCAL_CSV:
        if os.path.exists(s["path"]):
            with open(s["path"], encoding="utf-8") as fh:
                n = sum(1 for _ in fh) - 1
            digest = sha256_of(s["path"])
        else:
            n, digest = "", "(不在)"
        rows.append([os.path.basename(s["path"]), "受領 CSV", s["supplier"], s["as_of"],
                     n, digest, "", s["caveat"]])
    for a in sources.ARRIVED_NOT_INTEGRATED:
        rows.append([os.path.basename(a["path"]), "**届いているが未統合**",
                     "抽出側(um350)", as_of,
                     sum(1 for _ in open(a["path"], encoding="utf-8")) - 1
                     if os.path.exists(a["path"]) else "",
                     sha256_of(a["path"]) if os.path.exists(a["path"]) else "(不在)",
                     "", f'{a["what"]} —— {a["why_not_yet"]}'])
    for v in sources.VOLATILE:
        rows.append([v["column"], "**値が動く予定の列**", "抽出側(um350)", as_of,
                     v["n_affected_in_req3"], "", "yes",
                     f'{v["scope"]}: {v["what"]} —— {v["why"]}。{v["note"]}'])
    return write_csv(os.path.join(outdir, "provenance.csv"), header, rows)


def write_caveats(outdir: str):
    header = ["column", "answers", "not_answers", "trap", "by", "as_of", "detail"]
    rows = [[c["column"], c["answers"], c["not_answers"], c["trap"], c["by"],
             c["as_of"], c["detail"]] for c in CAVEATS]
    return write_csv(os.path.join(outdir, "caveats.csv"), header, rows)


def verify_conservation(outdir: str, expect_layers: int, expect_pages: int,
                        label: str) -> int:
    """**書き出した CSV を読み返して、総数が保存しているか確かめる。**

    抽出側の試走で、**層の ID だけを鍵にしたため別の頁の同じ ID が上書きされ、
    254層が160ファイルになった** —— **37% が黙って消え、エラーは 0**(2026-10-26、
    T109 の出力側の現れ)。**鍵が頁をまたいで一意でないと、出力が黙って減る。**

    だから在庫を信じず**読み返して足す**。メモリの中の数を足すと、
    書き出しで消えた分は見えない。
    """
    def total(fname, col):
        path = os.path.join(outdir, fname)
        with open(path, encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            i = header.index(col)
            return sum(int(r[i]) for r in reader if r[i].strip())

    def verdict(expect, got):
        """**文言を発火した条件から導く。**

        以前はどちらに外れても「消えている」と名乗り、**増えているときに
        「消えている: -1」と印字した**(2026-10-26)。
        **誤った欠陥を名乗る文言は、正しい欠陥を名乗ったものとして読まれる。**
        """
        if got == expect:
            return "一致"
        if got < expect:
            return f"**消えている: {expect - got}**(書き出しで落ちた)"
        return f"**増えている: {got - expect}**(二重に数えた)"

    got_layers = total("layers_by_year_kind_type.csv", "n_layers")
    got_pages = total("pages_by_year.csv", "n_pages")
    print(f"  [総数の保存: {label}] 読み返して足す(在庫ではなく出力を数える)")
    print(f"    層  入力 {expect_layers} → 出力 {got_layers}  "
          f"{verdict(expect_layers, got_layers)}")
    print(f"    頁  入力 {expect_pages} → 出力 {got_pages}  "
          f"{verdict(expect_pages, got_pages)}")
    return 0 if (got_layers == expect_layers and got_pages == expect_pages) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-out", default="census")
    ap.add_argument("--full-out", default="data/census_full")
    ap.add_argument("--req3", default=sources.by_key("req3_layer_kind")["path"])
    ap.add_argument("--added33", default=sources.by_key("added33_structure")["path"])
    ap.add_argument("--initcolor", default=sources.by_key("body_initcolor")["path"])
    ap.add_argument("--delivery-root", default=delivery.ROOT_DEFAULT)
    ap.add_argument("--sha256", default=sources.by_key("file_sha256")["path"])
    ap.add_argument("--faint", default=sources.by_key("req3_body_blocks")["path"])
    ap.add_argument("--gradmap", default=sources.by_key("grad_v1_v2_map")["path"])
    args = ap.parse_args()
    as_of = today()

    check_column_classes()
    ver = instrument_version()
    print(f"原稿の国勢調査 —— 走行 {as_of} / **器具の版 {ver['commit']}"
          f"(未コミットの変更: {ver['dirty']})**")
    if ver["dirty"] == "yes":
        print("  **器具に未コミットの変更が在る。この走行の数は、版では同定できない**"
              f"({ver['dirty_files']} ファイル)")
    print(f"  公開(段 A): {args.public_out}/   全量(段 C): {args.full_out}/")

    print("\n[材料を読む]")
    req3_rows, req3_pages, drops_r3 = load_req3(args.req3)
    print(f"  {args.req3}: {len(req3_rows)} 層 / {len(req3_pages)} 頁")
    drops_r3.report("依頼3")
    a33_rows, drops_a33 = load_added33(args.added33)
    print(f"  {args.added33}: {len(a33_rows)} 頁")
    drops_a33.report("added33")

    ic_rows, drops_ic, collapsed = load_initcolor(args.initcolor)
    print(f"  {args.initcolor}: {len(ic_rows)} 層")
    drops_ic.report("本体の InitColor")
    if collapsed:
        print(f"  [段 A で畳んだ `set`] {len(collapsed)} 個 → 「{fp.FILL_LABEL}」1行"
              f"(作品名を含むため)。畳んだ層 {sum(collapsed.values())}")
        print("    **段 C では畳まない**(世代もプールである)")

    drops_h = Drops()
    hash_rows = hashes.load(args.sha256, drops_h)
    print(f"  {args.sha256}: {len(hash_rows)} ファイル")
    drops_h.report("sha256")
    grad = hashes.grad_layer_map(args.gradmap, drops_h)
    print(f"  {args.gradmap}: {grad[1]} 層 / mask_cover が両方在る {grad[2]} / "
          f"**値が動いた {grad[3]}**")

    drops_f = Drops()
    faint_rows = faint.load(args.faint, drops_f)
    print(f"  {args.faint}: {len(faint_rows)} 層")
    drops_f.report("薄い画素の便")

    drops_dl = Drops()
    print(f"  {args.delivery_root}: 納品の木を歩く")

    overlap = req3_pages.keys() & {r["page_path"] for r in a33_rows}
    print(f"\n[母集団の重なり] 依頼3 ∩ added33 = {len(overlap)} 頁"
          f" → **別の母集団として別の行に出す**")

    layer_cells, page_cells = aggregate(req3_rows, req3_pages, a33_rows)
    print(f"[集計] 層のセル {len(layer_cells)} / 頁のセル {len(page_cells)}(1回だけ数える)")

    # 作品の異なり数は公開してよい(数であって名前ではない)。公開側に渡すのは**数だけ**
    n_works_by = collections.defaultdict(lambda: collections.Counter())
    seen = collections.defaultdict(set)
    for p in list(req3_pages.values()):
        if p["has_work"]:
            seen[("req3_layer_kind", p["year_label"])].add(p["work"])
    for r in a33_rows:
        if r["has_work"]:
            seen[("added33_structure", r["year_label"])].add(r["work"])
    for k, names in seen.items():
        n_works_by[k]["n_works"] = len(names)

    print("\n[公開(段 A)を書く] —— 行は PUBLIC_DIMS だけを持つ鍵から組む")
    os.makedirs(args.public_out, exist_ok=True)
    pub = emit_tables(args.public_out, layer_cells, page_cells, False, n_works_by)
    dl_pub, dl_rows, dl_collapsed, dl_names, dl_dup = emit_delivery(
        args.public_out, args.delivery_root, False, drops_dl)
    pub.update(dl_pub)
    print(f"  [納品の覆い] PNG {len(dl_rows)} 枚 / 名前の異なり {dl_names} / "
          f"**2つ以上の納品に同じ名前 {dl_dup}**")
    print(f"    畳んだ納品フォルダー {len(dl_collapsed)} 個(作品名を含むため)")
    drops_dl.report("納品の木")
    pub["req3_bound_classes.csv"] = emit_req3_bound_classes(
        args.public_out, faint_rows, req3_pages)
    pub["faint_alpha_gap.csv"] = emit_faint_gap(
        args.public_out, faint_rows, req3_pages)
    pub["occluder_classes.csv"] = emit_occluders(
        args.public_out, args.delivery_root, drops_dl)
    pub["occluder_balloon_two_sided.csv"] = emit_balloon_two_sided(
        args.public_out, args.delivery_root, drops_dl)
    pub["occluder_vocab_full.csv"] = emit_vocab_full(
        args.public_out, occluders.R7_DEFAULT, drops_dl)
    pub["occluder_front_position.csv"] = emit_front_position(
        args.public_out, occluders.R8_DEFAULT, drops_dl)
    pub["population_by_format.csv"] = emit_population_by_format(
        args.public_out, occluders.R8_DEFAULT, occluders.N1_DEFAULT, drops_dl)
    pub["occluder_folder_contents.csv"] = emit_folder_contents(
        args.public_out, occluders.R8_DEFAULT, drops_dl)
    pub["front_decile_kinds.csv"] = emit_front_decile_kinds(
        args.public_out, occluders.R8_DEFAULT, drops_dl)
    pub["coverage_bound_classes.csv"] = emit_bound_classes(
        args.public_out, ic_rows, False)
    pub.update(emit_duplicates(args.public_out, hash_rows, grad))
    pub["canvas_declared.csv"] = emit_canvas(args.public_out, a33_rows)
    pub.update(emit_full_population(args.public_out))
    pub["body_initcolor.csv"] = emit_initcolor(args.public_out, ic_rows, False)
    sha_by_path = {r["page_path"]: r["sha"] for r in hash_rows}
    pub["populations.csv"] = write_populations(args.public_out, req3_rows, req3_pages,
                                               a33_rows, as_of, sha_by_path)
    pub["provenance.csv"] = write_provenance(args.public_out, as_of)
    pub["caveats.csv"] = write_caveats(args.public_out)
    for name, n in sorted(pub.items()):
        print(f"    {n:6d} 行  {name}")
    rc_cons = verify_conservation(args.public_out, len(req3_rows),
                                  len(req3_pages) + len(a33_rows), "段 A")

    print("\n[全量(段 C)を書く] —— 同じ集計から")
    os.makedirs(args.full_out, exist_ok=True)
    full = emit_tables(args.full_out, layer_cells, page_cells, True, n_works_by)
    full.update(emit_delivery(args.full_out, args.delivery_root, True, Drops())[0])
    full["req3_bound_classes.csv"] = emit_req3_bound_classes(
        args.full_out, faint_rows, req3_pages)
    full["faint_alpha_gap.csv"] = emit_faint_gap(
        args.full_out, faint_rows, req3_pages)
    full["occluder_classes.csv"] = emit_occluders(
        args.full_out, args.delivery_root, Drops())
    full["occluder_balloon_two_sided.csv"] = emit_balloon_two_sided(
        args.full_out, args.delivery_root, Drops())
    full["occluder_vocab_full.csv"] = emit_vocab_full(
        args.full_out, occluders.R7_DEFAULT, Drops())
    full["occluder_front_position.csv"] = emit_front_position(
        args.full_out, occluders.R8_DEFAULT, Drops())
    full["population_by_format.csv"] = emit_population_by_format(
        args.full_out, occluders.R8_DEFAULT, occluders.N1_DEFAULT, Drops())
    full["occluder_folder_contents.csv"] = emit_folder_contents(
        args.full_out, occluders.R8_DEFAULT, Drops())
    full["front_decile_kinds.csv"] = emit_front_decile_kinds(
        args.full_out, occluders.R8_DEFAULT, Drops())
    full["coverage_bound_classes.csv"] = emit_bound_classes(
        args.full_out, ic_rows, True)
    full.update(emit_duplicates(args.full_out, hash_rows, grad))
    full["canvas_declared.csv"] = emit_canvas(args.full_out, a33_rows)
    full.update(emit_full_population(args.full_out))
    full["body_initcolor.csv"] = emit_initcolor(args.full_out, ic_rows, True)
    full["caveats.csv"] = write_caveats(args.full_out)
    for name, n in sorted(full.items()):
        print(f"    {n:6d} 行  {name}")
    rc_cons |= verify_conservation(args.full_out, len(req3_rows),
                                   len(req3_pages) + len(a33_rows), "段 C")

    rc_cons |= check_full_population_totals()
    rc_cons |= sources.audit_needs()
    rc_cons |= sources.audit_arrived()
    rc_cons |= audit_inline_needs(args.public_out)
    vocab = private_vocabulary(req3_pages, a33_rows)
    rc, _hits = audit_public(args.public_out, vocab)
    if rc:
        print("\n**固有名の指摘あり。この出力は commit しない**")
    elif rc_cons:
        print("\n**総数が保存していない。この出力は commit しない**")
    else:
        print("\n指摘なし(段 A に固有名は無く、総数も保存している)")
    return rc | rc_cons


if __name__ == "__main__":
    sys.exit(main())
