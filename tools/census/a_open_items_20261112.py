#!/usr/bin/env python
"""**a(決定論の除外)の残りの数え上げ** —— Track E の頼み(11-12 の4通)と Track Q の頼み1件。構造 DB だけで数える(画素は使わない)。

頼み → 節:
1. `tone_enabled`=1 と、E の表の 状態=確定 の行(名前・種別)との重なり。確定の行を全部当てたときの重なりを除いた層の数
2. フォルダーの中身(子孫すべて)の名前・種別・層の数: `外枠` / `tone` / `トーン` / `ネーム` / `"N"`(全頁)
3. 種別 `story_info` と `text` は CSP のどの属性から来ているか(`v_layer_kind` の定義と、両方の印を持つ層の数)
4. `.clip` の頁の数 / `.lip` の層の数 / そのうち `.clip` に同じ (名前, 種別) が無い層 / 作品単位
5. `断ち落とし枠`(raster)の構造: 親の名前・兄弟の名前と種別・`断ち落とし枠（白）` と同じ親に在るか
6. 同じ頁に `成人修正` の層が2つ以上ある頁の数(Q の頼み)
7. 層ごとの `a_verdict`(`template_excluded` / `not_excluded` / `undecided`)。**画素は使わない。空は別の列(`empty_c`)で、測っていない層は空欄**

**決めたこと(数える前に)**:
- 母集団は構造 DB の全層から恒久除外の頁の層を外したもの(436,425 + ... は印字で確かめる)
- E の表は読むだけ(sha256 を印字)。**状態が「確定」で始まり、決め方が 名前 / 種別 / 列 の行だけ**を a の除外に使う。「除外しない(確定)」は `not_excluded`
- フォルダーの行は**フォルダー自身だけ**(作者の裁定 11-12「7これもｑ」)。中身は数えるが除外にしない
- `text` 種別の層の名前は出さない(条例11)。名前の欄は `(text)` にする
- 出力は非公開(層名を含む)

    ./venv/bin/python tools/census/a_open_items_20261112.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402
from stage_hearing_round2_b import TABLE_V3, empty_c  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E_TABLE = os.path.join(os.path.dirname(ROOT), "lineart-aesthetic-judge", "config", "picture_exclusion_draft_20261111.tsv")
OUT_FOLDERS = "data/census_full/a_folder_contents_20261112.csv"
OUT_CROP = "data/census_full/a_crop_frame_structure_20261112.csv"
OUT_VERDICT = "data/census_full/a_verdict_20261112.csv"
FOLDERS = [("外枠", "folder"), ("tone", "folder"), ("トーン", "folder"), ("ネーム", "folder"), ('"N"', "folder")]
KIND_KEY = {"用紙": "paper", "(ノンブル)・(作品情報)": "story_info", "セリフのテキスト層": "text"}


def load_e():
    raw = open(E_TABLE, "rb").read()
    rows = [ln.split("\t") for ln in raw.decode("utf-8").splitlines() if ln and not ln.startswith("#")]
    head, body = rows[0], rows[1:]
    i_how, i_key, i_kind, i_state = head.index("決め方"), head.index("鍵"), head.index("種別"), head.index("状態")
    excl_name, excl_kind, keep_name = set(), set(), set()
    tone_col = False
    for r in body:
        how, key, kind, state = r[i_how], r[i_key], r[i_kind], r[i_state]
        if how == "除外しない" and state.startswith("除外しない(確定)"):
            keep_name.add((key, kind))
        elif state.startswith("確定"):
            if how == "名前":
                excl_name.add((key, kind))
            elif how == "種別":
                excl_kind.add(KIND_KEY[key])
            elif how == "列" and key.startswith("tone_enabled"):
                tone_col = True
    return hashlib.sha256(raw).hexdigest(), excl_name, excl_kind, tone_col, keep_name


def main() -> int:
    print("[a の残りの数え上げ(構造 DB だけ)]")
    sha, excl_name, excl_kind, tone_col, keep_name = load_e()
    print(f"    E の表 sha256 {sha[:16]} / 確定の名前の行 {len(excl_name)} / 確定の種別 {sorted(excl_kind)} / 列 tone_enabled {tone_col} / 除外しない(確定){len(keep_name)}")
    con = db.connect()
    ex = exclusions.load()
    un = split_units.units(con)
    rows = con.execute("""select l.page_path, l.layer_id, coalesce(l.name, ''), k.v_layer_kind, p.format, l.parent_id, l.order_in_parent, l.depth,
                                 coalesce(l.tone_enabled, false), l.layer_type, coalesce(l.has_text, false)
                            from layers l join v_layer_kind k using (page_path, layer_id) join pages p using (page_path)""").fetchall()
    n_ex = sum(1 for r in rows if r[0] in ex)
    rows = [r for r in rows if r[0] not in ex]
    print(f"    層 {len(rows) + n_ex:,} → 恒久除外の頁の層 {n_ex} を外して {len(rows):,}")
    by_id = {(r[0], r[1]): r for r in rows}

    # 1. 重なり
    print("\n  [1] tone_enabled=1 と確定の行の重なり")
    hit = collections.Counter()
    excl = {}
    for r in rows:
        why = []
        if (r[2], r[3]) in excl_name:
            why.append("名前:" + r[2])
        if r[3] in excl_kind:
            why.append("種別:" + r[3])
        if tone_col and r[8]:
            why.append("列:tone_enabled")
        if why:
            excl[(r[0], r[1])] = why
            if tone_col and r[8] and len(why) > 1:
                for w in why:
                    if w != "列:tone_enabled":
                        hit[w] += 1
    n_tone = sum(1 for r in rows if r[8])
    print(f"      tone_enabled=1 {n_tone:,} 層のうち、確定の名前・種別にも当たる層 {sum(1 for k, w in excl.items() if 'tone_enabled' in ' '.join(w) and len(w) > 1):,}")
    for w, n in hit.most_common():
        print(f"        {w}: {n:,}")
    multi = sum(1 for w in excl.values() if len(w) > 1)
    print(f"      **確定の行を全部当てて、重なりを1回だけ数えた層 {len(excl):,} / {len(rows):,} = {len(excl) / len(rows):.4f}**(2つ以上の行に当たる層 {multi:,})")

    # 2. フォルダーの中身
    print("\n  [2] フォルダーの中身(子孫すべて、全頁)")
    children = collections.defaultdict(list)
    for r in rows:
        children[(r[0], r[5])].append(r)
    out_f = []
    for fname, fkind in FOLDERS:
        roots = [r for r in rows if r[2] == fname and r[3] == fkind]
        cnt = collections.Counter()
        units = set()
        for f in roots:
            stack = [f]
            while stack:
                x = stack.pop()
                for c in children.get((x[0], x[1]), []):
                    nm = "(text)" if c[3] == "text" else c[2]
                    cnt[(nm, c[3])] += 1
                    stack.append(c)
            units.add(un.get(wd(f[0])) or wd(f[0]))
        tot = sum(cnt.values())
        n_ex_layers = sum(n for (nm, k), n in cnt.items() if (nm, k) in excl_name or k in excl_kind)
        print(f"      `{fname}`({fkind}): フォルダー {len(roots):,} / 作品単位 {len(units)} / 中の層 {tot:,} / 異なる(名前, 種別) {len(cnt)} / "
              f"そのうち確定の名前・種別に当たる層 {n_ex_layers:,}")
        for (nm, k), n in cnt.most_common(6):
            print(f"          {nm}({k}) {n:,}")
        for (nm, k), n in cnt.most_common():
            out_f.append(dict(folder=fname, folder_kind=fkind, n_folders=len(roots), child_name=nm, child_kind=k, n_layers=n,
                              in_confirmed=int((nm, k) in excl_name or k in excl_kind)))
    write(OUT_FOLDERS, out_f)

    # 3. story_info と text
    print("\n  [3] story_info と text の出どころ")
    si = [r for r in rows if r[3] == "story_info"]
    tx = [r for r in rows if r[3] == "text"]
    print(f"      story_info = layer_type 800(`has_text` より先に判定): {len(si):,} 層。うち has_text も真 {sum(1 for r in si if r[10]):,}")
    print(f"      text = has_text(layer_type 800 でない): {len(tx):,} 層。layer_type の内訳 " +
          " / ".join(f"{a} {b:,}" for a, b in collections.Counter(r[9] for r in tx).most_common(4)))
    print(f"      layer_type 800 で has_text が偽 {sum(1 for r in si if not r[10]):,}")

    # 4. .clip の頁と .lip の層
    print("\n  [4] .clip の頁 / .lip の層")
    clip_pages = {r[0] for r in rows if r[4] == "clip"}
    lip = [r for r in rows if r[4] == "lip"]
    clip_keys = {(r[2], r[3]) for r in rows if r[4] == "clip"}
    lip_only = [r for r in lip if (r[2], r[3]) not in clip_keys]
    allc = con.execute("select count(*) from pages where format = 'clip'").fetchone()[0]
    print(f"      .clip の頁 {len(clip_pages):,}(層の在る頁。pages 表の .clip {allc:,} から恒久除外 1 を外すと {allc - 1:,})")
    print(f"      .lip の層 {len(lip):,}(頁 {len({r[0] for r in lip}):,})/ そのうち .clip に同じ (名前, 種別) が1つも無い層 **{len(lip_only):,}**"
          f"(異なる (名前, 種別) {len({(r[2], r[3]) for r in lip_only}):,} / 作品単位 {len({un.get(wd(r[0])) or wd(r[0]) for r in lip_only})})")
    print(f"      .lip の層の作品単位 {len({un.get(wd(r[0])) or wd(r[0]) for r in lip})}")
    print(f"      見積もり: .clip {len(clip_pages):,} 頁 × 10.6 秒 = {len(clip_pages) * 10.6 / 3600:.1f} 時間(E の第3段の実測。最大 49 秒/頁)")

    # 5. 断ち落とし枠
    print("\n  [5] `断ち落とし枠`(raster)の構造")
    crop = [r for r in rows if r[2] == "断ち落とし枠" and r[3] == "raster"]
    par = collections.Counter()
    sib = collections.Counter()
    same_parent_white = 0
    out_c = []
    for r in crop:
        p = by_id.get((r[0], r[5]))
        pname = (p[2], p[3]) if p else ("(最上位)", "")
        par[pname] += 1
        sibs = [c for c in children.get((r[0], r[5]), []) if c[1] != r[1]]
        for c in sibs:
            sib[("(text)" if c[3] == "text" else c[2], c[3])] += 1
        w = any(c[2] == "断ち落とし枠（白）" for c in sibs)
        same_parent_white += w
        out_c.append(dict(page_path=r[0], layer_id=r[1], format=r[4], parent_name=pname[0], parent_kind=pname[1], depth=r[7],
                          order_in_parent=r[6], n_siblings=len(sibs), sibling_white_crop=int(w),
                          siblings="; ".join(sorted({f"{'(text)' if c[3] == 'text' else c[2]}|{c[3]}" for c in sibs}))))
    print(f"      層 {len(crop)}(形式 " + " / ".join(f"{a} {b}" for a, b in collections.Counter(r[4] for r in crop).items()) + ")")
    print("      親: " + " / ".join(f"{a[0]}({a[1]}) {b}" for a, b in par.most_common(5)))
    print("      兄弟の上位: " + " / ".join(f"{a[0]}({a[1]}) {b}" for a, b in sib.most_common(6)))
    print(f"      同じ親に `断ち落とし枠（白）` が在る {same_parent_white} / {len(crop)}")
    print("      深さ: " + " / ".join(f"{a} {b}" for a, b in sorted(collections.Counter(r[7] for r in crop).items())))
    write(OUT_CROP, out_c)

    # 6. 成人修正
    print("\n  [6] 同じ頁に `成人修正` が2つ以上")
    sj = collections.Counter(r[0] for r in rows if r[2] == "成人修正")
    fmt = {r[0]: r[4] for r in rows}
    ge2 = [p for p, n in sj.items() if n >= 2]
    print(f"      `成人修正` の層 {sum(sj.values())}(種別 " + " / ".join(f"{a} {b}" for a, b in collections.Counter(r[3] for r in rows if r[2] == '成人修正').items()) + ")"
          f" / 頁 {len(sj)} / **2つ以上の頁 {len(ge2)}**(clip {sum(1 for p in ge2 if fmt[p] == 'clip')} / lip {sum(1 for p in ge2 if fmt[p] == 'lip')})")
    print("      1頁あたりの数: " + " / ".join(f"{a}層 {b}頁" for a, b in sorted(collections.Counter(sj.values()).items())))

    # 7. a_verdict
    print("\n  [7] a_verdict(層ごと)")
    v3 = {(r["page_path"], int(r["layer_id"])): empty_c(r) for r in csv.DictReader(open(os.path.join(ROOT, TABLE_V3), encoding="utf-8"))}
    out_v = []
    cv = collections.Counter()
    ce = collections.Counter()
    for r in rows:
        k = (r[0], r[1])
        if k in excl:
            v = "template_excluded"
        elif (r[2], r[3]) in keep_name:
            v = "not_excluded"
        else:
            v = "undecided"
        e = v3.get(k, "")
        cv[v] += 1
        ce[e or "(空欄: 測っていない)"] += 1
        out_v.append(dict(page_path=r[0], layer_id=r[1], a_verdict=v, a_rule=" + ".join(excl.get(k, [])), empty_c=e))
    for v in ("template_excluded", "not_excluded", "undecided"):
        print(f"      {v}: {cv[v]:,} / {len(rows):,} = {cv[v] / len(rows):.4f}")
    print("      empty_c: " + " / ".join(f"{a} {b:,}" for a, b in ce.most_common()) + "(層の役割の表 v3 の層だけ測ってある)")
    write(OUT_VERDICT, out_v)
    return 0


def write(path, out):
    with open(os.path.join(ROOT, path), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"      → `{path}`(非公開)")


if __name__ == "__main__":
    raise SystemExit(main())
