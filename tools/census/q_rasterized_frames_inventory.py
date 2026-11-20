#!/usr/bin/env python
"""**ラスタ化したコマ枠の頁の棚卸し —— 既存の記録だけで**(Track Q の問い、2026-11-20)。新しい画素の列は作らない。

母集団(述語を数の隣に書く):
- **全数の側**: 構造 DB(10-26 の断面)の `.clip` の頁。恒久除外の頁は外して数を印字。`.lip` は画素が無いので数だけ
- **Q11 の側**: Q11 `_v2` の層の表(`q11_per_panel_layers_v2_20261120.csv`)の頁。`koma_note` で分ける
- **コマ枠のフォルダーが在る頁** = `v_layer_kind = frame_folder` の層が1つ以上(DB の `has_frame_line`)

問いと答えの出し方:
1. 時期(頁の年代札)× 作品フォルダーで、コマ枠のフォルダーが在る/無い頁
2. コマ枠のフォルダーが無い頁にある、**名前の候補**(層名・祖先のフォルダー名に 枠/コマ/waku/koma/frame。
   `モザイク` は別に数える)と、**紙の層の直上の層**(`z_global` で紙の次。フォルダーを除く)。
   **名前は候補で、正解ではない**(規律13 —— 的中率は出せない。ラスタ化した枠の正解の列が無いから)
3. コマ枠のフォルダーが在る頁で、**フォルダーの外**に名前の候補が在る頁
4. Q11 で「頁にコマの形が無い」頁の `koma_frames` の行(名前・誤り)

出力 `data/census_full/q_rasterized_frames_*_20261120.csv`(頁のパスを持つので公開しない・追跡しない)。
"""
import collections
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
Q11 = os.path.join(D, "q11_per_panel_layers_v2_20261120.csv")
OUT_PAGES = os.path.join(D, "q_rasterized_frames_pages_20261120.csv")
OUT_CAND = os.path.join(D, "q_rasterized_frames_name_candidates_20261120.csv")
PAT = [("枠", re.compile("枠")), ("コマ", re.compile("コマ|ｺﾏ")), ("koma", re.compile("koma", re.I)),
       ("waku", re.compile("waku", re.I)), ("frame", re.compile("frame", re.I)), ("ワク", re.compile("ワク"))]
MOSAIC = re.compile("モザイク")


def hits(n):
    return [k for k, rx in PAT if rx.search(n or "")]


def main() -> int:
    c = db.connect()
    ex = exclusions.load()
    pages = {p: (fmt, y) for p, fmt, y in c.execute("select page_path, format, year from pages").fetchall()}
    rows = c.execute("""select l.page_path, l.layer_id, coalesce(l.name,''), l.parent_id, l.z_global, l.folder,
                               k.v_layer_kind, l.blend_mode, l.visibility & 1
                          from layers l join v_layer_kind k using (page_path, layer_id)""").fetchall()
    by_page = collections.defaultdict(dict)
    for r in rows:
        by_page[r[0]][r[1]] = r
    fmt_n = collections.Counter(f for f, _ in pages.values())
    print(f"頁: {len(pages):,}(.clip {fmt_n['clip']:,} / .lip {fmt_n['lip']:,})")

    has_ff = {p for p, ls in by_page.items() if any(r[6] == "frame_folder" for r in ls.values())}
    print("コマ枠のフォルダーが在る頁(形式別): " + " / ".join(
        f"{f} {sum(pages[p][0] == f for p in has_ff):,}" for f in ("clip", "lip")))
    print("  .lip は画素が無いので以下は .clip だけ。.lip の在る頁は数えたが母集団に入れない")

    clip = sorted(p for p, (f, _) in pages.items() if f == "clip")
    n_ex = sum(p in ex for p in clip)
    clip = [p for p in clip if p not in ex]
    print(f".clip {fmt_n['clip']:,} 頁 − 恒久除外 {n_ex} = {len(clip):,} 頁")

    # Q11 の側
    q11 = collections.defaultdict(set)
    for r in csv.DictReader(open(Q11, encoding="utf-8")):
        note = r["koma_note"] or ("形を使えた" if r["koma_source"] else ("飛ばした" if r["reason"] else "表に無い"))
        q11[r["page_path"]].add(note)
    qcls = {}
    for p, s in q11.items():
        qcls[p] = ("列が無い" if any("列" in x for x in s) else "形が無い" if any("形が無い" in x for x in s)
                   else "表に無い" if "表に無い" in s else "形を使えた" if "形を使えた" in s else "飛ばしただけ")
    print(f"\nQ11 _v2 の頁: {len(q11):,} = " + " / ".join(f"{k} {v:,}" for k, v in collections.Counter(qcls.values()).most_common()))
    miss = [p for p in q11 if p not in pages]
    ffq = collections.Counter((qcls[p], p in has_ff) for p in q11 if p in pages)
    print(f"  DB に無い頁 {len(miss)}。Q11 の区分 × DB でコマ枠のフォルダーが在るか:")
    for (k, h), v in sorted(ffq.items()):
        print(f"    {v:6,d}  {k} × {'在る' if h else '無い'}")

    # 1. 時期 × 作品
    tab = collections.defaultdict(lambda: [0, 0, 0, 0])  # 在る, 無い, Q11列無し, Q11形あり
    for p in clip:
        y = pages[p][1]
        k = (y, wd(p))
        tab[k][0 if p in has_ff else 1] += 1
        if qcls.get(p) == "列が無い":
            tab[k][2] += 1
        if qcls.get(p) == "形を使えた":
            tab[k][3] += 1
    by_y = collections.defaultdict(lambda: [0, 0, 0, 0])
    for (y, _), v in tab.items():
        for i in range(4):
            by_y[y][i] += v[i]
    print("\n1. 年代札ごと(.clip、恒久除外の後、頁): コマ枠のフォルダー 在る / 無い | Q11 の 列が無い / 形を使えた")
    for y in sorted(by_y):
        a = by_y[y]
        print(f"  {y:8s} {a[0]:6,d} / {a[1]:6,d} | {a[2]:5,d} / {a[3]:5,d}")
    nw = collections.Counter((p in has_ff) for p in clip)
    print(f"  計 {nw[True]:,} / {nw[False]:,}")
    with open(OUT_PAGES, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["year_label", "work_dir", "pages_frame_folder", "pages_no_frame_folder",
                    "q11_pages_no_frame_column", "q11_pages_panels_used"])
        for (y, wdir), v in sorted(tab.items()):
            w.writerow([y, wdir] + v)
    mixed = sum(1 for (y, wdir), v in tab.items() if v[0] and v[1])
    print(f"  作品フォルダー {len(tab)}: 在るだけ {sum(1 for v in tab.values() if v[0] and not v[1])} / "
          f"無いだけ {sum(1 for v in tab.values() if v[1] and not v[0])} / 両方の頁を持つ {mixed}")

    # 2・3. 名前の候補
    cand = []
    pat_n = collections.defaultdict(lambda: collections.Counter())
    pat_p = collections.defaultdict(lambda: collections.defaultdict(set))
    above_paper = collections.Counter()
    above_paper_pages = collections.defaultdict(set)
    z_none = [0]
    cls_n = collections.Counter()
    cls_p = collections.defaultdict(set)
    for p in clip:
        ls = by_page.get(p, {})
        ff = p in has_ff
        side = "フォルダー在る頁" if ff else "フォルダー無い頁"
        in_ff = {}

        def anc(lid):
            out = []
            r = ls.get(lid)
            seen = 0
            while r and r[3] is not None and r[3] in ls and seen < 50:
                r = ls[r[3]]
                out.append(r)
                seen += 1
            return out
        for lid, r in ls.items():
            if r[5]:
                continue
            a = anc(lid)
            under_ff = any(x[6] == "frame_folder" for x in a)
            if under_ff:
                continue
            own = hits(r[2])
            an = sorted({h for x in a for h in hits(x[2])})
            mos = bool(MOSAIC.search(r[2])) or any(MOSAIC.search(x[2]) for x in a)
            if own or an:
                key = ("層名 " + "+".join(own)) if own else ("祖先 " + "+".join(an))
                if mos:
                    key += "(モザイク)"
                pat_n[side][key] += 1
                pat_p[side][key].add(p)
                cls = ("断ち落とし(テンプレート)" if "断ち落とし" in r[2] or any(x[2] == "外枠" for x in a)
                       else "テキストの層" if r[6] == "text" else "コマの指定の書き込み" if "指定" in r[2]
                       else "名前の候補")
                cls_n[(side, cls)] += 1
                cls_p[(side, cls)].add(p)
                cand.append(dict(cls=cls, page_path=p, layer_id=lid, page_side=side, layer_name=r[2],
                                 ancestors="/".join(x[2] for x in reversed(a)), own_hits="+".join(own),
                                 ancestor_hits="+".join(an), mosaic=int(mos), v_layer_kind=r[6],
                                 blend_mode=r[7], self_shown=r[8]))
        z_none[0] += sum(1 for r in ls.values() if not r[5] and r[4] is None)
        leaves = sorted((r for r in ls.values() if not r[5] and r[4] is not None), key=lambda r: r[4])
        for i, r in enumerate(leaves[:-1]):
            if r[6] == "paper":
                nx = leaves[i + 1]
                k = (side, nx[6], "名前に候補" if hits(nx[2]) else "名前に候補なし")
                above_paper[k] += 1
                above_paper_pages[k].add(p)
    for side in ("フォルダー無い頁", "フォルダー在る頁"):
        print(f"\n{'2' if side == 'フォルダー無い頁' else '3'}. {side}(コマ枠のフォルダーの外の層、フォルダーの層を除く): 名前の型 → 層 / 頁")
        for k, v in pat_n[side].most_common():
            print(f"  {v:6,d} / {len(pat_p[side][k]):5,d}  {k}")
        allp = set().union(*pat_p[side].values()) if pat_p[side] else set()
        nomos = set().union(*(s for k, s in pat_p[side].items() if "モザイク" not in k)) if pat_p[side] else set()
        print(f"  頁(重なりを除く): {len(allp):,}(モザイクの型を除くと {len(nomos):,})")
    print("\n名前の型を中身で分けた(層 / 頁):")
    for k, v in sorted(cls_n.items()):
        print(f"  {v:6,d} / {len(cls_p[k]):5,d}  {k[0]} / {k[1]}")
    print(f"\n(z_global が無い層 {z_none[0]:,} は直上の判定から外した)")
    print("紙の層の直上の層(z_global で紙の次、フォルダーを除く): 頁の側 × 種別 × 名前 → 層")
    for k, v in sorted(above_paper.items(), key=lambda x: -x[1])[:20]:
        print(f"  {v:6,d}  {k[0]} / {k[1]} / {k[2]}")
    with open(OUT_CAND, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cand[0])); w.writeheader(); w.writerows(cand)

    # 4. 形が無い頁
    sh = [p for p, k in qcls.items() if k == "形が無い"]
    kf = collections.defaultdict(list)
    for p, n, err in c.execute("select page_path, name, error from koma_frames").fetchall():
        kf[p].append((n or "", err))
    kinds = collections.Counter()
    for p in sh:
        es = kf.get(p, [])
        if not es:
            kinds["koma_frames に行が無い"] += 1
        elif all(MOSAIC.search(n) for n, _ in es):
            kinds["行は全部 モザイク"] += 1
        elif all(err for n, err in es if not MOSAIC.search(n)):
            kinds["モザイク以外の行は全部 誤り(10-26 の断面)"] += 1
        else:
            kinds["モザイク以外の読める行が在る(断面の版の違い?)"] += 1
    print(f"\n4. Q11 で「頁にコマの形が無い」頁 {len(sh)}: " + " / ".join(f"{k} {v}" for k, v in kinds.most_common()))
    print(f"\n出力: {OUT_PAGES}\n      {OUT_CAND}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
