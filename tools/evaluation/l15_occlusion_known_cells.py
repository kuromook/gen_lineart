# -*- coding: utf-8 -*-
"""遮蔽の既知セル。**O1 / O2 / O2' / O3 を回す前に回す。落ちたらどれも読まない**(規律2)。

登録は `doc/work_log.md` の 2026-10-26 18:09 の節(KC56)と 2026-10-27 18:41 の節(作者の定義)。

**KC56 は納品を一切使わない。** 画布も遮蔽物もマスクも**ここで作る** ——
T132(**既知セルには、試験対象の器具が作っていない入力が少なくとも1つ必要**)を
満たしている唯一の形である。**こちらが持っていた KC53 は「前と同じ器具か」しか見ない。**

> **KC53 は「前と同じ器具か」を見る。KC56 は「器具が正しいか」を見る。**

**規則を書き写さない。** 試験するのは本体と同じ関数
(`l15_zorder_occlusion.front_by_z` / `occluder_cover`)である。
**書き写せば、書き写した側が正しいことしか示せない。**

見るもの(登録どおり):

1. **重なる遮蔽物**(手前)で `|R ∧ OCC_front| / |R|` が**手で決めた既知の値**になること
2. **隣に置いた遮蔽物**(重なりなし、手前)で **0.0000** になること
3. **後ろの遮蔽物**は重なっていても**数に入らない**こと(z の向き)
4. **マスクを掛けた後の形が、手で決めた形と画素単位で一致**すること(O3 の経路)
5. **読み込みで落ちた遮蔽物が理由別に数えられる**こと(規律9。黙って消えないこと)
"""
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from l15_zorder_occlusion import front_by_z, occluder_cover

H = W = 400
TMP = None   # kc56 の中で作る(`TMPDIR` を尊重する一時ディレクトリ。/tmp を決め打ちしない)


def png(name, mask, shape=(H, W)):
    """`mask` が True の画素を**墨(0)**、残りを白(255)にして書く。器具は `<128` を墨と読む。"""
    im = np.full(shape, 255, np.uint8)
    im[mask] = 0
    p = os.path.join(TMP, name)
    cv2.imwrite(p, im)
    return p


def kc56():
    global TMP
    TMP = tempfile.mkdtemp(prefix='l15_kc56_')
    R = np.zeros((H, W), bool); R[100:200, 100:200] = True        # 面 10,000px、z=10
    nR = int(R.sum())

    a = np.zeros((H, W), bool); a[100:140, 100:200] = True        # R と 4,000px 重なる
    b = np.zeros((H, W), bool); b[100:200, 250:300] = True        # 隣。重なり 0
    c = np.zeros((H, W), bool); c[160:200, 100:200] = True        # R と 4,000px 重なるが **後ろ**
    d = np.zeros((H, W), bool); d[100:200, 100:200] = True        # R と全面(10,000px)重なる
    mk = np.zeros((H, W), bool); mk[100:200, 100:150] = True      # **<128 が見せる側**。左半分だけ

    pa, pb, pc, pd = png('a.png', a), png('b.png', b), png('c.png', c), png('d.png', d)
    pm = png('m.png', mk)
    psmall = png('small.png', np.ones((50, 50), bool), (50, 50))  # 形が違う
    pmiss = os.path.join(TMP, 'nope.png')                         # 読めない

    ok = True

    def cover(occ, want, label):
        nonlocal ok
        front, drop = front_by_z(occ, [10], (H, W))
        got = int((R & front[10]).sum()) / nR
        good = abs(got - want) < 1e-12
        ok &= good
        print(f'  {label:44} {got:.4f}  (= {want:.4f} であるべき)  '
              f'{"PASS" if good else "**FAIL**"}')
        return front, drop

    print('KC56 合成の遮蔽(**納品を一切使わない。T132 を満たす**)')
    print(f'  面 R = {nR:,}px  z=10。遮蔽物は z を明示して置く')
    cover([(1, pa, 20)], 0.4000, '1. 重なる遮蔽物(手前 z=20、4,000px)')
    cover([(2, pb, 30)], 0.0000, '2. 隣の遮蔽物(手前 z=30、重なり 0)')
    cover([(3, pc, 5)], 0.0000, '3. **後ろ**の遮蔽物(z=5、4,000px 重なる)')
    cover([(1, pa, 20), (3, pc, 5)], 0.4000, '4. 手前 + 後ろ(後ろは数に入らない)')
    cover([(1, pa, 20), (2, pb, 30)], 0.4000, '5. 手前2枚(隣は足さない)')

    # ---- 4. マスクを掛けた後の形が、手で決めた形と**画素単位で**一致するか
    ink, why = occluder_cover(pd, pm, (H, W))
    same = ink is not None and bool((ink == (d & mk)).all())
    print(f'  6. マスクを掛けた後の形が手で決めた形と画素単位で一致      '
          f'{"**一致**" if same else "**違う** " + str(why)}  '
          f'{"PASS" if same else "**FAIL**"}')
    ok &= same
    cover([(4, pd, 20, pm)], 0.5000,
          '7. マスク付きの遮蔽物(全面 10,000 の左半分だけ見せる)')
    cover([(4, pd, 20)], 1.0000, '8. 同じ層をマスク無しで(全面)')

    # ---- 5. 落ちた遮蔽物が理由別に数えられるか(**黙って消えないこと**)
    front, drop = front_by_z([(5, pmiss, 20), (6, psmall, 20), (1, pa, 20)],
                             [10], (H, W))
    got = int((R & front[10]).sum()) / nR
    want_drop = {'遮蔽物の PNG が読めない': 1, '遮蔽物の形が頁と違う': 1}
    good = dict(drop) == want_drop and abs(got - 0.4000) < 1e-12
    ok &= good
    print(f'  9. 落ちた遮蔽物を理由別に数える: {dict(drop)}  '
          f'(= {want_drop} であるべき。残り1枚で {got:.4f})  '
          f'{"PASS" if good else "**FAIL**"}')
    _, why2 = occluder_cover(pd, psmall, (H, W))
    good2 = why2 == 'マスクの形が頁と違う'
    ok &= good2
    print(f'  10. マスクの形が違うときの理由: {why2!r}  {"PASS" if good2 else "**FAIL**"}')

    shutil.rmtree(TMP, ignore_errors=True)
    print(f'  **KC56 {"PASS" if ok else "FAIL"}**')
    return ok


def main():
    print('L15 遮蔽の既知セル (**1つでも落ちたら O1 / O2 / O2\' / O3 を読まない**)')
    ok = kc56()
    print()
    print('**既知セル PASS。遮蔽の腕を回してよい**' if ok else
          '**落ちた。遮蔽の腕を回さない**')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
