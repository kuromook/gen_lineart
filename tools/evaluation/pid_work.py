# -*- coding: utf-8 -*-
"""`page_id` から作品を決める。**既定値に落とさず、分からなければ落とす。**

2026-10-26 の点検で立てた。器具のあちこちに

    w = 'ako6' if '_ako6_' in pid else 'ako7'

と書いてあった。**これは二択の落とし方で、`ako6` でないものを全部 `ako7` にする** ——
`ako7` かどうかを一度も確かめていない。**沈黙する既定値**(罠 T104 の族)である。

**今日の突き合わせでは食い違い0**だった(65頁の `work` 列と、7,362層の経路、
どちらとも一致)。**だから直すのは、間違っていたからではなく、
間違ったときに黙るからである。**

あわせて作者の言明(2026-10-26、M 経由):
**`pageNNNN` は両方向に外れる** —— 漫画の頁が `pageNNNN` でないことがあり
(`done*` がまさにそれ)、`pageNNNN` が漫画の頁でないこともある。
**この65頁では `pageNNNN` が素40頁と `done*` 25頁を完全に分ける**が、
**それは偶然の一致であって規則ではない。** 現行かどうかは**フォルダー**で見る
(作者が「ここでは別の意味」と明言したのは `2018/ako6` と `2019/ako7` の下だけで、
`_bk` / `backup` / `Done` / `rev2` には広げない)。
"""
WORKS = ('ako6', 'ako7')


def work_of(pid, path=None):
    """`pid`(と在れば経路)から作品を返す。**決まらなければ ValueError。**"""
    if path:
        parts = path.replace('\\', '/').split('/')
        for w in WORKS:
            if w in parts:
                return w
    hit = [w for w in WORKS if f'_{w}_' in pid]
    if len(hit) == 1:
        return hit[0]
    raise ValueError(
        f'作品が決まらない: pid={pid!r} path={path!r} 当たり={hit} —— '
        '**既定値に落とさない。** 新しい作品なら WORKS に足すこと')


def is_current_page(path):
    """現行の頁か。**`done*` は作業途中の退避**(作者 2026-10-26:
    「全部同じ運用だ　共通して背景がない」)。**頁名ではなくフォルダーで見る。**"""
    return not path.replace('\\', '/').split('/')[-2].startswith('done')
