# -*- coding: utf-8 -*-
"""**追跡されている画像を数え、許可表に無いものが在れば終了コード 1。**

作者の裁定(2026-10-26):

> **いくらかはサンプルとして画像は載せたい。でないと何のプロジェクトか分からない。
> 但しこちらが意図しない出方はだめだ。当面は見つかり次第対処とする。**

**だから「画像を追跡しない」ではなく「意図した見本だけを追跡する」である。**
判断の線は**人が選んで置いたか、器具が書いて紛れ込んだか**で、
`ALLOWED` がその「選んだ」の実体である。**表に無い画像が在れば止まる。**

**なぜ `.gitignore` では足りないか**(2026-10-26 に確かめた):
`results/**` は 2026-07-11 から ignore されているのに、
**原寸の切り出し2枚が 2026-10-23 に `git add -f` で入っていた。**

> **ignore は、既に追跡されているファイルも `-f` も止めない。**
> **止めるのは終了コードである。**

**T133(点検が自分を指した件)をここで踏まないために**: この表には
**頁番号や作品の識別子を持つ名前を一切書かない。** 書けば、
**守る対象を点検自身が運ぶことになる。** 表に載せられない名前の画像は、
**そもそも追跡してはいけないものである**(名前がそう言っている)。

使い方:  ./venv/bin/python tools/check_public_images.py
  終了コード 0 = 表のとおり / 1 = 表に無い画像が追跡されている
"""
import os
import subprocess
import sys

EXT = ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tif', '.tiff')

# **人が選んで置いた見本だけ。** 増やすのは人の仕事で、器具の仕事ではない。
ALLOWED = {
    # 何のプロジェクトかを示す説明図(作者の「サンプルとして載せたい」に当たる)
    'results/explain/a_raster.png',
    'results/explain/b_production.png',
    'results/explain/c_missing.png',
    'results/explain/raster_vs_production_fill.png',
    # 試験入力(縮小済み。原寸ではない)
    'test/rough/a.jpg',
    'test/rough/sample.jpg',
    'test/rough/sample01.jpg',
    'test/rough/sample_face.jpg',
}


def tracked_images():
    out = subprocess.run(['git', 'ls-files', '-z'], capture_output=True,
                         check=True, cwd=os.path.dirname(os.path.dirname(
                             os.path.abspath(__file__))))
    files = out.stdout.decode('utf-8').split('\0')
    return sorted(f for f in files if f.lower().endswith(EXT))


def main():
    found = tracked_images()
    extra = [f for f in found if f not in ALLOWED]
    gone = sorted(ALLOWED - set(found))

    print(f'追跡されている画像 {len(found)}  /  許可表 {len(ALLOWED)}')
    # **落としたものではなく、通したものも数える**(規律9 を両側に)
    print(f'  表のとおり {len(found) - len(extra)}')
    if gone:
        # **消えていても止めない。** 消すのは人の判断で、これは漏れの点検である
        print(f'  表に在るが追跡されていない {len(gone)}(**止めない**。消すのは人の判断)')
        for f in gone:
            print(f'    {f}')
    if extra:
        print(f'  **表に無い {len(extra)}**')
        for f in extra:
            print(f'    {f}')
        print()
        print('**追跡された画像が許可表に無い。**'
              ' 見本として載せるなら ALLOWED に足す(人が選ぶ)。'
              ' 器具が書いたものなら `git rm --cached` で外す。')
        print('**`.gitignore` は `-f` も既追跡も止めない。'
              ' だからここで終了コード 1 にする。**')
        return 1
    print('-> 表のとおり。**意図しない出方は無い**')
    return 0


if __name__ == '__main__':
    sys.exit(main())
