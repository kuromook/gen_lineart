# 「完全に空」の定義を測る(作者、2026-11-11「画素無くてもマスクがあればその層は空ではない
# 完全に空であるという定義が必要」)。
# 名前は出さない(条例11)。束・項目・位置・種別だけ印字する。
import csv, collections, hashlib, io, os, sys

D = "data/role_predictor"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()[:16]

tab_p = os.path.join(D, "layer_role_table_20261110_v2.csv")
key_p = os.path.join(D, "hearing_round2_a_key_20261110.csv")
print("表 :", tab_p, "sha256", sha(tab_p))
print("鍵 :", key_p, "sha256", sha(key_p))

rows = list(csv.DictReader(io.open(tab_p, encoding="utf-8")))
print("表の行:", len(rows))

MASK_COLS = ["has_mask", "has_own_mask", "n_ancestor_masks", "mask_status",
             "n_px_a_gt0_own_mask", "n_px_a_gt0_all_masks", "empty"]

def i(r, c):
    v = r.get(c, "")
    return None if v in ("", None) else int(v)

# ---- 1. 既に在る `empty` 列は何で決まっているか
print("\n== 1. 表の `empty` 列と画素の列の関係(落としていない。全53,289行) ==")
cross = collections.Counter()
for r in rows:
    px = i(r, "n_px_a_gt0")
    cross[(r.get("empty", ""), "px=0" if px == 0 else ("px>0" if px else "px空欄"))] += 1
for k in sorted(cross, key=str): print(f"  empty={k[0]!r:8} {k[1]:8} {cross[k]:>7}")

# ---- 2. 画素0の層が、マスクを持っているか
print("\n== 2. `n_px_a_gt0 == 0` の層を、マスクの列で割る ==")
z = [r for r in rows if i(r, "n_px_a_gt0") == 0]
print("  画素0の層:", len(z))
for c in MASK_COLS:
    cnt = collections.Counter(r.get(c, "") for r in z)
    print(f"  {c:22}", dict(sorted(cnt.items(), key=str)))

print("\n== 3. 候補の定義ごとに、画素0の層が何層「空でない」になるか ==")
def defs(r):
    """候補の定義。**画素の列が空欄の層は、どの定義でも「空」にしない**(測れない。
    render_status が「描画できなかった」の層。空であることを示していない)。"""
    px   = i(r, "n_px_a_gt0")
    own  = i(r, "n_px_a_gt0_own_mask")
    allm = i(r, "n_px_a_gt0_all_masks")
    ho   = r.get("has_own_mask", "")
    nam  = i(r, "n_ancestor_masks")
    measurable = px is not None
    z = measurable and px == 0
    no_own = ho in ("0", "", "False", "false") or own in (0,)
    return {
      "定義A 画素0だけ(前の読み)": z,
      "定義B +自分のマスクが無い/空": z and no_own,
      "定義C +マスク越しの画素も0": z and no_own and (allm in (0, None)),
      "定義D +祖先のマスクも無い(作者の言に最も近い)": z and no_own and (allm in (0, None)) and (nam in (0, None)),
    }
tally = collections.Counter(); tot = collections.Counter()
for r in rows:
    for k, v in defs(r).items():
        tot[k] += 1
        if v: tally[k] += 1
for k in sorted(tally): print(f"  {k:34} {tally[k]:>7} 層 / 53,289")

# ---- 4. 便A の97行で、束ごとに
print("\n== 4. 便A の97行。定義ごとの「空」の数(束別) ==")
key = list(csv.DictReader(io.open(key_p, encoding="utf-8")))
tb = {(r["page_path"], r["layer_id"]): r for r in rows}
per = collections.defaultdict(lambda: collections.Counter())
miss = 0
detail = []
for k in key:
    t = tb.get((k["page_path"], k["layer_id"]))
    if t is None: miss += 1; continue
    dd = defs(t)
    per[k["bundle"]]["行"] += 1
    for name, v in dd.items():
        if v: per[k["bundle"]][name] += 1
    if dd["定義A 画素0だけ(前の読み)"]:
        detail.append((k["bundle"], k["item"], k["pos"], k["kind"], t))
print("  照合不能:", miss)
names = list(defs(rows[0]).keys())
print("  | 束 | 行 | " + " | ".join(n.split()[0] for n in names) + " |")
print("  |---|---:|" + "---:|" * len(names))
for b in sorted(per):
    print(f"  | {b} | {per[b]['行']} | " + " | ".join(str(per[b][n]) for n in names) + " |")
print(f"  | 計 | {sum(per[b]['行'] for b in per)} | " +
      " | ".join(str(sum(per[b][n] for b in per)) for n in names) + " |")

print("\n== 5. 画素0の5行を、マスクの列で1行ずつ(層名は書かない。条例11) ==")
for b, it, pos, kind, t in detail:
    f = " ".join(f"{c}={t.get(c,'')!r}" for c in MASK_COLS)
    print(f"  {b} 項目{it} 位置{pos} {kind}: {f}")

print("\n== 6. ラベルの在る層(作者の段)で、画素0の層の段 ==")
lab = [r for r in rows if r.get("author_stage", "") not in ("", None)]
print("  ラベルの在る層:", len(lab))
zl = [r for r in lab if i(r, "n_px_a_gt0") == 0]
print("  うち画素0:", len(zl))
c = collections.Counter(r["author_stage"] for r in zl)
print("  段の内訳:", dict(c))
for name in names[1:]:
    sub = [r for r in zl if defs(r)[name]]
    cc = collections.Counter(r["author_stage"] for r in sub)
    print(f"  {name:34} に残る: {len(sub):>3} 層  段 {dict(cc)}")
