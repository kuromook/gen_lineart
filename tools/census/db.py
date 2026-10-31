#!/usr/bin/env python
"""構造 DB の断面を**読み取り専用で**開く。**開く前に、同じ断面かを確かめる。**

**census は読むだけ。書き戻さない**(ユーザー、2026-10-28)。
**DB 自身は作業木の外**に置く —— `git` に巻き込まれる経路を無くすため
(2026-10-23 に 99MB を public repo に上げた事故がある。force-push では消えない)。
控えは `doc/db_snapshot.md`。

**この器具が確かめるのは、開いた断面が控えと同じファイルかである。**
**届かないもの**: **DB の中身の正しさ。** 断面の数は**抽出側が測ったもの**で、
ここで測り直したのではない(条例5: 供給元から取った数は測った者の名を帯びる)。
**便の CSV と食い違ったら、便が正本**(`doc/db_snapshot.md`)。
"""
import hashlib
import os

PATH = "/home/sh1/deepl/census_db/manuscripts.duckdb"
# **控えの値**(`doc/db_snapshot.md`)。**ここを書き換えて通すのではなく、
# 断面が変わったら控えを書き直してから、この2つを直す**
BYTES = 157_298_688
SHA256 = "65434b73af6c91ca8e6a50b1436419e40f3fae6e67424ddf28b9c4ee4c77bbb9"
AS_OF = "2026-10-26 17:57"          # 供給元のファイル日時(JST)
SUPPLIER = "抽出側(um350)"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def connect(verify: bool = True):
    """**読み取り専用で開く。** 断面が控えと違えば、開かずに落ちる。"""
    import duckdb
    if not os.path.exists(PATH):
        raise SystemExit(f"**構造 DB が無い**: {PATH}(控えは `doc/db_snapshot.md`)")
    n = os.path.getsize(PATH)
    if verify:
        if n != BYTES:
            raise SystemExit(
                f"**断面が違う** —— 大きさ {n} バイト、控えは {BYTES}。"
                "**控え(`doc/db_snapshot.md`)を書き直してから `db.py` を直すこと**")
        got = sha256_of(PATH)
        if got != SHA256:
            raise SystemExit(
                f"**断面が違う** —— sha256 {got[:16]}…、控えは {SHA256[:16]}…")
        print(f"    構造 DB の断面: {AS_OF}(供給元 {SUPPLIER})"
              f" / {n} バイト / sha256 一致")
    return duckdb.connect(PATH, read_only=True)
