# -*- coding: utf-8 -*-
"""基盤の1問: **`cov_honest` は標準の「分割」に効くのか、「墨の和集合」だけに効くのか。**

**答: 分割に効く。しかも強く、`conflict` の規則を通して効く。**
これを**合成の既知セル(KC48)で示す** —— 本物の原稿の主値を1つも見ずに答えられる。

機序(`l18_propagate.one_page()` の中):

    s, L, adj = seed_of(R, lab, P)       # ← **面ごとに1つの層ラベル L**
    C, ids    = prop_full(s, cl)
    owner[cid].append((gi, L, adj))
    ...
    cf = int(len({t[1] for t in lst}) > 1)   # ← **同じ成分を別の L が主張したら競合**
    if cf:   conf_px += ar                   # ← **競合した成分は cov_honest に入らない**
    elif pu >= 0.90: honest += ar

`seed_of` は**面の境界から距離 d 以内の線の墨の argmax ラベル**を返す。だから:

- **面を割ると、割れ目に新しい境界ができ、各破片が自分の L を持つ**
- **同じ線の成分を別の L が主張すれば `cf = 1` になり、その成分は `cov_honest` から落ちる**
- **分割が細かくなるほど主張者が増え、`cov_honest` は下がる**

**墨の和集合が同じでも値が変わる** —— それを下で示す。

**帰結**: 対照の移動の単位だけを粗くすると、**量そのものが変わる**(別の分割に対する
`cov_honest` になる)。**だから「同じ量の安い推定量」はこの軸には無い。**
費用を下げるなら**分割を変えない形**で下げる必要がある(§2 の計測)。
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import A
from l17_known_cells import assign_top
from l18_known_cells import comp_pur, ink_components, prop_full, seed_of

H = W = 400


def honest_of(regions, lab, cl, st_, P):
    """`one_page` の採点部をそのまま小さく再現する。**規則は書き写さず同じ関数を呼ぶ。**"""
    owner = {}
    for R in regions:
        s, L, adj = seed_of(R, lab, P)
        if adj == 0 or L == 0:
            continue
        C, ids = prop_full(s, cl)
        for x in ids:
            owner.setdefault(int(x), []).append((L, adj))
    honest = conf = 0
    detail = []
    for cid, lst in owner.items():
        lst.sort(key=lambda t: -t[1])
        L = lst[0][0]
        ar = int(st_[cid, A])
        cf = int(len({t[0] for t in lst}) > 1)
        pu = comp_pur(cl == cid, lab, L)
        if cf:
            conf += ar
        elif pu >= 0.90:
            honest += ar
        detail.append((cid, ar, sorted({t[0] for t in lst}), round(pu, 4), cf))
    return honest, conf, detail


def kc48():
    """**KC48**: 墨の和集合を**1画素も変えずに**標準の分割だけを変えると、
    `cov_honest` の分子が **面積ぶん → 0** に落ちる。

    合成: 1つの線の成分が、**ほぼ全部が層1・端の小片だけが層2**でできている。

      版A(割らない): 面は1つ。境界は層1にだけ接する -> 主張者は L=1 だけ
                      -> 競合なし、純度 >= 0.90 -> **honest に入る**
      版B(割る)    : 同じ領域を2つに割り、片方の境界が層2の小片に接する
                      -> 主張者が L=1 と L=2 -> **競合 -> honest から落ちる**
    """
    P = np.ones((H, W), bool)
    # ---- 線の墨: 1成分。層1 が大部分、層2 が端の小片(接しているので1成分)
    l1 = np.zeros((H, W), bool); l1[190:210, 60:300] = True      # 20 x 240
    l2 = np.zeros((H, W), bool); l2[190:210, 300:312] = True     # 20 x 12 (端で接する)
    lab, keys = assign_top([(1, l1, 1), (2, l2, 2)], (H, W))
    ink, cl, st_, n = ink_components(lab, P)
    print(f'KC48 線の墨: 成分 {n - 1} 個  '
          f'(1個であるべき)  層1 {int(l1.sum()):,}px  層2 {int(l2.sum()):,}px  '
          f'層2 の割合 {l2.sum() / (l1.sum() + l2.sum()):.4f}')
    if n - 1 != 1:
        print('     **合成が壊れている(成分が1個でない)。この点検は成立しない**')
        return False

    # ---- 版A: 割らない1つの面(層1 の下に置き、境界が層1 に接する)
    rA = np.zeros((H, W), bool); rA[212:330, 60:300] = True   # 境界が墨の 3px 下 (D1=4 の内)
    hA, cA, dA = honest_of([rA], lab, cl, st_, P)
    # ---- 版B: 同じ和集合を2つに割る(右の破片の境界が層2 の小片に接する)
    rB1 = np.zeros((H, W), bool); rB1[212:330, 60:296] = True
    rB2 = np.zeros((H, W), bool); rB2[212:330, 298:312] = True
    hB, cB, dB = honest_of([rB1, rB2], lab, cl, st_, P)

    print(f'     版A 面1つ : honest {hA:,}  競合 {cA:,}  内訳 {dA}')
    print(f'     版B 面2つ : honest {hB:,}  競合 {cB:,}  内訳 {dB}')
    ok = hA > 0 and hB == 0 and cB > 0
    print(f'     **分割だけを変えて honest が {hA:,} -> {hB:,} '
          f'(墨の和集合は1画素も変えていない)**  {"PASS" if ok else "FAIL"}')
    if ok:
        print('     -> **`cov_honest` は標準の分割に依存する。** '
              '対照の移動の単位を粗くすると量が変わる')
    else:
        print('     -> **示せていない。** 合成を作り直す必要がある'
              '(この点検が落ちた場合、分割依存の主張は根拠を失う)')
    return ok


def main():
    print('L35 分割感度 —— 基盤の1問への答(**合成だけ。本物の主値は見ない**)')
    ok = kc48()
    print()
    print('**結論: `cov_honest` は分割に依存する**' if ok else
          '**結論を出せない。合成が示せていない**')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
