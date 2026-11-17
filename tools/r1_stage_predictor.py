# -*- coding: utf-8 -*-
"""登録 R1(config/prereg_stage_predictor_r1_20261112.md、追補1〜3)の走行器具。

評価 = 便A の主 86行(92行から 2024\\mayer を外し、分からないを外した行)。
学習 = 12頁265層 + 便B の作者の答え。評価と作品フォルダーが重なる行は学習から外す。
腕 A0〜A5、IC1〜IC3、NC1〜NC3、C1〜C3 を登録のとおりに印字する。層名は印字しない。
使い方: ./venv/bin/python tools/r1_stage_predictor.py
"""
import csv, collections, hashlib, io, math, random, sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GroupKFold
from scipy.sparse import hstack, csr_matrix

ANS_A = "data/census/hearing_round2_a_answers.csv"
ANS_B = "data/census/hearing_round2_b_answers.csv"
ROLE = "data/census/layer_role_table_20261112_v3.csv"
EXCL = "../lineart-census/data/census_full/author_exclusions.csv"
ORDER = ("あたり", "下絵", "清書", "段ではない")
STAGE_A = {"あたり": "あたり", "下絵": "下絵", "清書(線画)": "清書",
           "段の線ではない(塗り・効果・資料など)": "段ではない",
           "テンプレート(トンボ・裁ち落とし枠など、印刷のための層)": "段ではない",
           "分からない": "分からない"}
STAGE_B = dict(STAGE_A)
STAGE_B.update({"テンプレート": "段ではない", "グレー彩色": "段ではない", "ベタ": "段ではない", "陰影": "段ではない",
                "段の線ではない(上のどれでもない塗り・効果・資料など)": "段ではない"})
NUM = ["layer_type", "colour_type_index", "self_shown_db", "ancestors_shown_db", "opacity", "blend_mode", "clip_raw",
       "has_mask", "tone_enabled", "pos_from_top", "n_leaf_layers", "n_px_a_gt0", "n_px_a_ge26",
       "n_px_a_gt0_own_mask", "n_px_a_gt0_all_masks", "has_own_mask", "n_ancestor_masks",
       "all_masks_from_koma_vector", "draw_color_enable"]
FORBIDDEN = ("author_", "why_listed", "textish", "page_path", "work_dir", "split_unit", "year_label", "empty_c",
             "canvas_px")
GRID = [(n, d) for n in (50, 100, 200) for d in (1, 2, 3)]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def is3d_line(c):
    return bool(c) and ("3D" in c or "３D" in c or "３Ｄ" in c) and "線画" in c


def load():
    rt = {(r["page_path"], r["layer_id"]): r for r in csv.DictReader(io.open(ROLE, encoding="utf-8"))}
    excl = set()
    try:
        for r in csv.DictReader(io.open(EXCL, encoding="utf-8")):
            excl.add(r.get("page_path") or r.get("path") or "")
    except FileNotFoundError:
        print("**恒久除外の名簿が読めない: %s —— 止める**" % EXCL)
        sys.exit(1)
    return rt, excl


def build_eval(rt, excl):
    a = list(csv.DictReader(io.open(ANS_A, encoding="utf-8")))
    rows, drop = [], collections.Counter()
    for r in a:
        k = (r["page_path"], r["layer_id"])
        t = rt[k]
        st = STAGE_A[r["author_stage"]]
        rows.append({"key": k, "t": t, "stage": st, "empty": r["empty"] == "1",
                     "mayer": "\\2024\\mayer\\" in r["page_path"], "excl": r["page_path"] in excl,
                     "unit": t["split_unit"], "src": "A"})
    return rows


def build_train(rt, excl, eval_units):
    drop = collections.Counter()
    out = {}
    remap3d = 0
    for k, t in rt.items():
        s = t["author_stage"]
        if not s:
            continue
        if s not in ORDER:
            drop["265層: 語彙の外(%s)" % s] += 1
            continue
        st = s
        if is3d_line(t["author_comment"]) and s == "清書":
            st = "段ではない"
            remap3d += 1
        out[k] = {"key": k, "t": t, "stage": st, "src": "265", "ver": "265", "comment": t["author_comment"]}
    n265 = len(out)
    composite = 0
    vers = collections.Counter()
    for r in csv.DictReader(io.open(ANS_B, encoding="utf-8")):
        k = (r["page_path"], r["layer_id"])
        if r["answer_source"] != "author":
            drop["便B: 機械の行"] += 1
            continue
        st = STAGE_B.get(r["stage"])
        if st is None:
            print("**便B の写せない選択肢: %s —— 止める**" % r["stage"])
            sys.exit(1)
        if st == "分からない":
            drop["便B: 分からない"] += 1
            continue
        if k in out:
            drop["265層: 便B と同じ層(便B を残す)"] += 1
        c = r["author_comment"]
        if c in ("ベタでできた陰影", "背景およびベタ"):
            composite += 1
        out[k] = {"key": k, "t": rt[k], "stage": st, "src": "B", "ver": r["sheet_version"], "comment": c}
    rows = []
    for k, r in out.items():
        t = r["t"]
        if r["key"][0] in excl:
            drop["%s: 恒久除外の頁" % r["src"]] += 1
            continue
        if t["empty_c"] == "1":
            drop["%s: 空(empty_c=1)" % r["src"]] += 1
            continue
        if t["split_unit"] in eval_units:
            drop["%s: 評価と作品フォルダーが重なる(%s)" % (r["src"], t["split_unit"])] += 1
            continue
        r["unit"] = t["split_unit"]
        rows.append(r)
        vers[(r["src"], r["ver"])] += 1
    return rows, drop, remap3d, composite, vers


def text_of(t):
    return (t["name"] or "") + " | " + (t["ancestors"] or "")


def numeric(t):
    w, h = fnum(t["canvas_px_w"]), fnum(t["canvas_px_h"])
    area = w * h if w and h else float("nan")
    x0, y0, x1, y1 = (fnum(t[c]) for c in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"))
    v = [fnum(t[c]) for c in NUM]
    pos, nl = fnum(t["pos_from_top"]), fnum(t["n_leaf_layers"])
    v.append(pos / nl if nl else float("nan"))
    v.append(fnum(t["n_px_a_gt0"]) / area if area else float("nan"))
    v.append((x1 - x0) / w if w else float("nan"))
    v.append((y1 - y0) / h if h else float("nan"))
    v.append((x1 - x0) * (y1 - y0) / area if area else float("nan"))
    v.append(1.0 if t["render_status"] == "描いた" else 0.0)
    return v


NUM_NAMES = NUM + ["pos_ratio", "drawn_frac", "bbox_w_frac", "bbox_h_frac", "bbox_area_frac", "render_status=描いた"]
CATS = {"name_rule": ("清書", "下絵", "あたり", "なし"), "v_layer_kind": None}


class Feats:
    def __init__(self, train, kind):
        self.kind = kind
        self.vec = CountVectorizer(analyzer="char", ngram_range=(1, 2), min_df=2, binary=True)
        self.vec.fit([text_of(r["t"]) for r in train])
        self.vlk = sorted({r["t"]["v_layer_kind"] for r in train})

    def X(self, rows):
        txt = self.vec.transform([text_of(r["t"]) for r in rows])
        if self.kind == "A4":
            return txt.toarray()
        nr = [[1.0 if r["t"]["name_rule"] == v else 0.0 for v in CATS["name_rule"][0:4]] for r in rows]
        vk = [[1.0 if r["t"]["v_layer_kind"] == v else 0.0 for v in self.vlk] for r in rows]
        num = np.nan_to_num(np.array([numeric(r["t"]) for r in rows]), nan=-1.0)
        return np.hstack([txt.toarray(), np.array(nr), np.array(vk), num])

    def names(self):
        n = ["ngram:" + g for g in self.vec.get_feature_names_out()]
        if self.kind == "A4":
            return n
        return n + ["name_rule=" + v for v in CATS["name_rule"]] + ["v_layer_kind=" + v for v in self.vlk] + NUM_NAMES


def fit_gb(Xtr, ytr, groups, label):
    k = min(5, len(set(groups)))
    best = None
    for n, d in GRID:
        accs = []
        for tr, va in GroupKFold(n_splits=k).split(Xtr, ytr, groups):
            m = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=0.1, random_state=0)
            m.fit(Xtr[tr], ytr[tr])
            accs.append((m.predict(Xtr[va]) == ytr[va]).mean())
        s = float(np.mean(accs))
        if best is None or s > best[0]:
            best = (s, n, d)
    print("   %s: 学習の側の交差検証(GroupKFold %d)で選んだ n_estimators=%d max_depth=%d(CV 的中 %.4f。棒ではない)"
          % (label, k, best[1], best[2], best[0]))
    m = GradientBoostingClassifier(n_estimators=best[1], max_depth=best[2], learning_rate=0.1, random_state=0)
    m.fit(Xtr, ytr)
    return m


def sign_p(a_hits, b_hits):
    """a が b に勝つかの片側符号検定。片方だけ当たった行だけを数える。"""
    w = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
    l = sum(1 for x, y in zip(a_hits, b_hits) if y and not x)
    n = w + l
    p = sum(math.comb(n, i) for i in range(w, n + 1)) / 2 ** n if n else 1.0
    return w, l, p


def report(name, pred, ev):
    y = [r["stage"] for r in ev]
    hits = [p == s for p, s in zip(pred, y)]
    return hits, sum(hits) / len(y)


def per_stage(pred, ev):
    y = [r["stage"] for r in ev]
    out = []
    for s in ORDER:
        na = sum(1 for v in y if v == s)
        npred = sum(1 for p in pred if p == s)
        tp = sum(1 for p, v in zip(pred, y) if p == s and v == s)
        prec = "%.4f" % (tp / npred) if npred else "判定保留(母数0)"
        miss = "%.4f" % ((na - tp) / na) if na else "判定保留(母数0)"
        out.append("%s 的中 %s / 取りこぼし %s(作者 %d・予測 %d)" % (s, prec, miss, na, npred))
    return out


def main():
    print("答え A : %s sha256 %s" % (ANS_A, sha(ANS_A)))
    print("答え B : %s sha256 %s" % (ANS_B, sha(ANS_B)))
    print("役割表 : %s sha256 %s" % (ROLE, sha(ROLE)))
    print("除外名簿: %s sha256 %s" % (EXCL, sha(EXCL)))
    rt, excl = load()
    allA = build_eval(rt, excl)
    if any(r["excl"] for r in allA):
        print("**評価に恒久除外の頁が在る: %d 行**" % sum(r["excl"] for r in allA))
    pops = {
        "主(92行 − 2024\\mayer − 分からない = 86)": [r for r in allA if not r["empty"] and not r["mayer"] and r["stage"] in ORDER],
        "元の87行(92行 − 分からない)": [r for r in allA if not r["empty"] and r["stage"] in ORDER],
        "97行の側(空を入れる、分からないを外す)": [r for r in allA if r["stage"] in ORDER],
    }
    main_ev = pops["主(92行 − 2024\\mayer − 分からない = 86)"]
    eval_units = {r["unit"] for r in allA}
    train, drop, remap3d, composite, vers = build_train(rt, excl, eval_units)

    print("\n== 学習の行 ==")
    print("   残った行 %d。出所と版: %s" % (len(train), dict(vers)))
    print("   段: %s" % dict(collections.Counter(r["stage"] for r in train)))
    print("   **落とした行(規律9、理由別)**:")
    for k, v in sorted(drop.items()):
        print("     %-50s %d" % (k, v))
    print("   3D の線画を 清書 → 段ではない に写した 265層の行(追補2): %d" % remap3d)
    print("   合成と読めるコメントの便B の行(そのまま使う、追補3): %d" % composite)
    print("   学習の作品フォルダー: %d" % len({r["unit"] for r in train}))

    # IC3
    ov = {r["unit"] for r in train} & eval_units
    print("\n== IC3 学習と評価の作品フォルダーの重なり: %d ==" % len(ov))
    if ov:
        print("**重なりが在る。止める**: %s" % ov)
        return 1

    ytr = np.array([r["stage"] for r in train])
    groups = np.array([r["unit"] for r in train])
    maj = collections.Counter(ytr).most_common(1)[0][0]
    nashi = [r["stage"] for r in train if r["t"]["name_rule"] == "なし"]
    fill = collections.Counter(nashi).most_common(1)[0][0]
    print("\n   学習の多数派 = %s / 学習で name_rule=なし の行の多数派 = %s(%d 行)" % (maj, fill, len(nashi)))

    # A3 頻度表
    def a3key(t):
        anc = (t["ancestors"] or "").lower()
        return (t["name_rule"], t["blend_mode"], any(w in anc for w in ("sketch", "下絵", "ラフ")))
    tab = collections.defaultdict(collections.Counter)
    for r in train:
        tab[a3key(r["t"])][r["stage"]] += 1

    def arms_pred(ev):
        p = {}
        p["A0 多数派"] = [maj] * len(ev)
        p["A1 名前の当て方"] = [r["t"]["name_rule"] if r["t"]["name_rule"] != "なし" else "なし" for r in ev]
        p["A2 名前+学習の多数派"] = [r["t"]["name_rule"] if r["t"]["name_rule"] != "なし" else fill for r in ev]
        a3 = []
        for r, a2 in zip(ev, p["A2 名前+学習の多数派"]):
            c = tab.get(a3key(r["t"]))
            a3.append(c.most_common(1)[0][0] if c else a2)
        p["A3 頻度表"] = a3
        return p

    print("\n== 学習(A4・A5・NC)==")
    f4, f5 = Feats(train, "A4"), Feats(train, "A5")
    # NC3
    bad = [n for n in f5.names() if not n.startswith("ngram:") and any(n.startswith(b) for b in FORBIDDEN)]
    print("   NC3 A5 の n-gram 以外の列(%d): %s" % (len([n for n in f5.names() if not n.startswith('ngram:')]),
                                             [n for n in f5.names() if not n.startswith("ngram:")]))
    print("   NC3 n-gram の数: %d / 禁止の列: %s" % (sum(1 for n in f5.names() if n.startswith("ngram:")), bad or "無し"))
    if bad:
        return 1
    m4 = fit_gb(f4.X(train), ytr, groups, "A4")
    m5 = fit_gb(f5.X(train), ytr, groups, "A5")
    # NC1: 作品フォルダーの中で段を並べ替える
    rng = random.Random(0)
    ysh = ytr.copy()
    for u in set(groups):
        idx = [i for i, g in enumerate(groups) if g == u]
        vals = [ysh[i] for i in idx]
        rng.shuffle(vals)
        for i, v in zip(idx, vals):
            ysh[i] = v
    m_nc1 = GradientBoostingClassifier(n_estimators=m5.n_estimators, max_depth=m5.max_depth, learning_rate=0.1,
                                       random_state=0).fit(f5.X(train), ysh)
    # NC2: 頁と作品の指紋だけ
    years = sorted({r["t"]["year_label"] for r in train} | {r["t"]["year_label"] for r in allA})

    def fp(rows):
        return np.array([[fnum(r["t"]["canvas_px_w"]), fnum(r["t"]["canvas_px_h"]), fnum(r["t"]["n_leaf_layers"])]
                         + [1.0 if r["t"]["year_label"] == y else 0.0 for y in years] for r in rows])
    m_nc2 = fit_gb(np.nan_to_num(fp(train), nan=-1.0), ytr, groups, "NC2(指紋だけ)")

    for label, ev in pops.items():
        print("\n" + "=" * 72)
        print("== 母集団: %s —— %d 行 ==" % (label, len(ev)))
        print("   作者の答え: %s" % dict(collections.Counter(r["stage"] for r in ev)))
        p = arms_pred(ev)
        p["A4 名前だけのモデル"] = list(m4.predict(f4.X(ev)))
        p["A5 本命"] = list(m5.predict(f5.X(ev)))
        p["NC1 並べ替えた段で学習"] = list(m_nc1.predict(f5.X(ev)))
        p["NC2 指紋だけ"] = list(m_nc2.predict(np.nan_to_num(fp(ev), nan=-1.0)))
        speaks = [r["t"]["name_rule"] != "なし" for r in ev]
        hits = {}
        print("   | 腕 | 全行 | 名前が言う行 | 「なし」の行 |")
        print("   |---|---:|---:|---:|")
        for k, pr in p.items():
            h, acc = report(k, pr, ev)
            hits[k] = h
            hs = [x for x, s in zip(h, speaks) if s]
            hn = [x for x, s in zip(h, speaks) if not s]
            print("   | %s | %d/%d = %.4f | %d/%d = %.4f | %d/%d = %.4f |" % (
                k, sum(h), len(h), acc, sum(hs), len(hs), sum(hs) / len(hs) if hs else float("nan"),
                sum(hn), len(hn), sum(hn) / len(hn) if hn else float("nan")))
        print("\n   段ごと(A5 本命):")
        for s in per_stage(p["A5 本命"], ev):
            print("     " + s)
        print("   段ごと(A2 名前+学習の多数派):")
        for s in per_stage(p["A2 名前+学習の多数派"], ev):
            print("     " + s)
        print("\n   符号検定(片側、片方だけ当たった行)")
        for other in ("A1 名前の当て方", "A2 名前+学習の多数派", "A3 頻度表", "A4 名前だけのモデル"):
            w, l, pv = sign_p(hits["A5 本命"], hits[other])
            print("     A5 対 %s: A5 だけ当たり %d / 相手だけ当たり %d → p = %.4f" % (other, w, l, pv))
        print("\n   作品フォルダーごと(A5 / A2 / 行数)")
        byu = collections.defaultdict(list)
        for r, a5, a2 in zip(ev, hits["A5 本命"], hits["A2 名前+学習の多数派"]):
            byu[r["unit"]].append((a5, a2))
        for u, v in sorted(byu.items(), key=lambda x: -len(x[1])):
            print("     %-28s 行 %2d  A5 %2d  A2 %2d" % (u, len(v), sum(a for a, _ in v), sum(b for _, b in v)))
        nc = [i for i, r in enumerate(ev) if r["unit"] != "2017\\copy"]
        if len(nc) != len(ev):
            print("   2017\\copy を除いた %d 行(副の印字、棒ではない): A5 %d / A2 %d / A1 %d / A0 %d" % (
                len(nc), sum(hits["A5 本命"][i] for i in nc), sum(hits["A2 名前+学習の多数派"][i] for i in nc),
                sum(hits["A1 名前の当て方"][i] for i in nc), sum(hits["A0 多数派"][i] for i in nc)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
