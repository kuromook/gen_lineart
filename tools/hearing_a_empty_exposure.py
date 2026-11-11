# A-2 に添えられた一言(「空の層は『分からない』のままで構いません」)の曝露を数える。
# 曝露 = その束の中で、1画素も描かれていない層の行。
import csv, collections, hashlib, os, sys
D = "data/role_predictor"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()[:16]

key_p = os.path.join(D, "hearing_round2_a_key_20261110.csv")
tab_p = os.path.join(D, "layer_role_table_20261110_v2.csv")
print("鍵 :", key_p, "sha256", sha(key_p))
print("表 :", tab_p, "sha256", sha(tab_p))

key = list(csv.DictReader(open(key_p, encoding="utf-8")))
print("鍵の行:", len(key))

tab = {}
with open(tab_p, encoding="utf-8") as f:
    r = csv.DictReader(f)
    cols = r.fieldnames
    for row in r:
        tab[(row["page_path"], row["layer_id"])] = row
print("表の行:", len(tab))
px_col = "n_px_a_gt0"
assert px_col in cols, [c for c in cols if "px" in c]

per = collections.defaultdict(lambda: {"行": 0, "空": 0, "照合不能": 0, "画素あり": 0})
empties = []
for k in key:
    b = k["bundle"]
    s = per[b]; s["行"] += 1
    t = tab.get((k["page_path"], k["layer_id"]))
    if t is None or t[px_col] == "":
        s["照合不能"] += 1
        continue
    if int(t[px_col]) == 0:
        s["空"] += 1
        empties.append((b, k["item"], k["pos"], k["kind"]))
    else:
        s["画素あり"] += 1

print()
print("| 束 | 行 | 画素あり | 空(0画素) | 照合不能 |")
print("|---|---:|---:|---:|---:|")
tot = collections.Counter()
for b in sorted(per):
    s = per[b]
    print(f"| {b} | {s['行']} | {s['画素あり']} | {s['空']} | {s['照合不能']} |")
    for k2, v in s.items(): tot[k2] += v
print(f"| 計 | {tot['行']} | {tot['画素あり']} | {tot['空']} | {tot['照合不能']} |")
print()
print("空の層の在る束と項目(層名は書かない。条例11):")
for b, it, pos, kind in empties:
    print(f"  {b} 項目{it} 位置{pos} 種別{kind}")
print()
print("添えた一言が触れる行 = その束の空の行。A-1 以外の束の空の行の数:",
      sum(1 for b, *_ in empties if b != "A-1"))
