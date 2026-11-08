# -*- coding: utf-8 -*-
"""L39 NC3: 作品を跨いで、学習頁と試験頁が**同じ層 PNG(バイト一致)**を持っていないか。

登録 §5 NC3。L24 の共有(`ako6 done4/ako6-18 ↔ ako7 page0073`)は `done*` なので学習に入らないはず。
**現行の素の頁どうし**で、line 側・fill 側の層 PNG の md5 を作品ごとに集めて交差を数える。
"""
import collections
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from l17_known_cells import arms, jobs, zmap                       # noqa: E402
from l38_merge_cause_20261106 import drop_wip                      # noqa: E402


def main():
    fill, line = arms('gated')
    cut = collections.Counter()
    js = drop_wip(jobs(zmap(), fill, line), cut)
    h = collections.defaultdict(dict)
    for pid, work, fz, lz, sel in js:
        for lid, path in list(fz) + [(l, p) for l, p, _ in lz]:
            if os.path.exists(path):
                h[work].setdefault(hashlib.md5(open(path, 'rb').read()).hexdigest(), []).append(
                    (pid, lid, os.path.getsize(path)))
    both = set(h['ako6']) & set(h['ako7'])
    print(f'NC3 層 PNG の md5: ako6 {len(h["ako6"])} 種 / ako7 {len(h["ako7"])} 種 / '
          f'**作品を跨いで一致 {len(both)}**')
    for k in sorted(both, key=lambda k: -h['ako6'][k][0][2])[:20]:
        print(f'  {h["ako6"][k][0]}  <->  {h["ako7"][k][0]}')


if __name__ == '__main__':
    main()
