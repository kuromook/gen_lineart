#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""白い層の例外の聞き取りの一覧を作る(2026-11-22、ユーザー「OK 聞き取りしよう」)。

第1層(10行) = 白さ >= 0.5 で、作者が『あたり・下絵・清書』と答えた層(反証が発火した層そのもの)。
第2層(10行、任意) = 名前が `white` でなく a がまだ落としていない、白さ >= 0.99 の層。
  —— 規則が当たる相手そのもの。第1層だけだと「落としてはいけない層」しか聞けない。

出力は data/ 配下(原稿由来。コミットしない)。層名は入れない(条例11 と同じ作法で、
画面に出すかは通達で別に論じる)。
"""
import csv, glob, collections, os, random

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def I(v):
    return int(v) if v not in ("", None) else None

def load():
    W = {}
    for f in sorted(glob.glob(BASE + "/data/census/white_column_layers_b*_20261122.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            W[(r["page_path"], r["layer_id"])] = r
    FLAG = {}
    for r in csv.DictReader(open(BASE + "/data/census/e_white_name_flag_20261122.csv", encoding="utf-8")):
        FLAG[(r["page_path"], r["layer_id"])] = r["name_white_raster"]
    V3 = {}
    for r in csv.DictReader(open(BASE + "/inbox/a_verdict_v3_20261115.csv", encoding="utf-8")):
        V3[(r["page_path"], r["layer_id"])] = r["a_verdict_v3"]
    BB = {}
    for r in csv.DictReader(open(BASE + "/inbox/full_clip_pixels_layers_merged.csv", encoding="utf-8")):
        BB[(r["page_path"], r["layer_id"])] = r
    ROLE = {}
    for r in csv.DictReader(open(BASE + "/inbox/layer_role_table_20261112_v3.csv", encoding="utf-8")):
        ROLE[(r["page_path"], r["layer_id"])] = r
    return W, FLAG, V3, ROLE, BB

def bbox_of(BB, k):
    o = BB.get(k, {})
    d = {c: o.get(c, "") for c in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")}
    try:
        w = int(d["bbox_x1"]) - int(d["bbox_x0"])
        h = int(d["bbox_y1"]) - int(d["bbox_y0"])
        d["bbox_w"], d["bbox_h"] = w, h
        # 原寸1:1 の切り出しは 1600px 窓。bbox が収まるなら bbox のまま
        d["crop_rule"] = "bbox そのまま" if max(w, h) <= 1600 else "bbox の中心に 1600x1600 の窓"
    except (ValueError, TypeError):
        d["bbox_w"] = d["bbox_h"] = ""
        d["crop_rule"] = "bbox が無い(頁の合成だけ)"
    return d


def year_of(page_path):
    parts = page_path.replace("/", "\\").split("\\")
    return parts[2] if len(parts) > 2 else ""

def main():
    W, FLAG, V3, ROLE, BB = load()
    rows = []

    # ---- 第1層: 反証が発火した層(作者ラベル在り) ----
    first = []
    for k, r in W.items():
        if r["drawn_as"] != "raster":
            continue
        nd = I(r["n_drawn"])
        if not nd:
            continue
        ro = ROLE.get(k)
        if not ro or ro.get("author_stage", "") not in ("あたり", "下絵", "清書"):
            continue
        wf = I(r["n_white"]) / nd
        if wf < 0.50:
            continue
        first.append((wf, k, r, ro))
    first.sort(reverse=True)

    def category(r, ro, wf):
        c = ro.get("author_comment", "")
        if "フキダシ" in c:
            return "フキダシ"
        if "描き文字" in c:
            return "描き文字"
        if "3D" in c or "３D" in c:
            return "3D の背景線画"
        if wf >= 0.999 and I(r["black_opaque"]) == 0:
            return "全部白い清書"
        return "その他"

    for wf, k, r, ro in first:
        cat = category(r, ro, wf)
        rows.append(dict(
            stratum="1:反証が発火した層", page_path=k[0], layer_id=k[1],
            white_frac=f"{wf:.4f}", n_drawn=r["n_drawn"], n_white=r["n_white"],
            black_opaque=r["black_opaque"], colour_opaque=r["colour_opaque"],
            grey_mid=r["grey_mid"], canvas_px=r["canvas_px"],
            coverage=f"{I(r['n_drawn'])/I(r['canvas_px']):.4f}",
            author_stage_before=ro["author_stage"], author_comment_before=ro["author_comment"],
            e_category=cat, a_v3=V3.get(k, ""), name_white_raster=FLAG.get(k, ""),
            year_folder=year_of(k[0]),
            **bbox_of(BB, k),
            question="この層は『白い塗りの層』として絵から除外してよいか(どの頁でも)",
        ))

    # ---- 第2層: 規則が当たる相手(名前 white でなく、a がまだ落としていない) ----
    cand = []
    for k, r in W.items():
        if r["drawn_as"] != "raster":
            continue
        nd = I(r["n_drawn"])
        if not nd:
            continue
        if FLAG.get(k) == "1":
            continue
        if V3.get(k) == "template_excluded":
            continue
        wf = I(r["n_white"]) / nd
        if wf < 0.99:
            continue
        cand.append((k, r, wf, nd))
    # 年のフォルダーごとに分けて、各年から大きい順に取る(年代はプールである)
    by_year = collections.defaultdict(list)
    for k, r, wf, nd in cand:
        by_year[year_of(k[0])].append((nd, k, r, wf))
    for y in by_year:
        by_year[y].sort(reverse=True)
    # 20行の上限が効くので、候補の多い年から10年を取る(規則が当たる場所で聞く)。
    # 取らなかった年は印字する —— 年代はプールである(基本法 §3)。
    order = sorted(by_year, key=lambda y: -len(by_year[y]))
    take, skip = order[:10], order[10:]
    print("第2層に入れなかった年(20行の上限のため):",
          {y: len(by_year[y]) for y in sorted(skip)})
    # 年ごとに**最大ではなく中位の覆い率**の層を取る —— 最大を取ると全面白の層ばかりになり、
    # 母集団(覆い率の中央値は下に印字)と形が違う標本になる。
    picked = []
    for y in sorted(take):
        v = sorted(by_year[y], key=lambda t: I(t[2]["n_drawn"]) / I(t[2]["canvas_px"]))
        picked.append((y, v[len(v) // 2]))
    cov_pop = sorted(I(r["n_drawn"]) / I(r["canvas_px"]) for k, r, wf, nd in cand)
    print("第2層の母集団 %d層 の覆い率: 中央値 %.4f / p25 %.4f / p75 %.4f / 全面(>=0.99) %d層"
          % (len(cov_pop), cov_pop[len(cov_pop) // 2], cov_pop[len(cov_pop) // 4],
             cov_pop[len(cov_pop) * 3 // 4], sum(1 for c in cov_pop if c >= 0.99)))
    for y, (nd, k, r, wf) in picked:
        rows.append(dict(
            stratum="2:規則が当たる相手(任意)", page_path=k[0], layer_id=k[1],
            white_frac=f"{wf:.4f}", n_drawn=r["n_drawn"], n_white=r["n_white"],
            black_opaque=r["black_opaque"], colour_opaque=r["colour_opaque"],
            grey_mid=r["grey_mid"], canvas_px=r["canvas_px"],
            coverage=f"{nd/I(r['canvas_px']):.4f}",
            author_stage_before="", author_comment_before="",
            e_category="名前の無い白い層", a_v3=V3.get(k, ""), name_white_raster="0",
            year_folder=y,
            **bbox_of(BB, k),
            question="この層は『白い塗りの層』として絵から除外してよいか(どの頁でも)",
        ))

    cols = ["pos", "stratum", "page_path", "layer_id", "year_folder", "white_frac", "coverage",
            "n_drawn", "n_white", "black_opaque", "colour_opaque", "grey_mid", "canvas_px",
            "author_stage_before", "author_comment_before", "e_category", "a_v3",
            "name_white_raster", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
            "bbox_w", "bbox_h", "crop_rule", "question"]
    out = BASE + "/data/role_predictor/white_exception_hearing_list_20261122.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for n, row in enumerate(rows, 1):
            row["pos"] = n
            w.writerow(row)
    print("wrote", out, len(rows), "rows")
    c = collections.Counter(r["stratum"] for r in rows)
    print(dict(c))
    print("第1層の区分:", dict(collections.Counter(r["e_category"] for r in rows if r["stratum"].startswith("1"))))
    print("第2層の年:", [r["year_folder"] for r in rows if r["stratum"].startswith("2")])
    print("第2層の候補の総数(名前 white でなく a 未落下・白さ>=0.99):", len(cand))

if __name__ == "__main__":
    main()
