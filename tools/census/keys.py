#!/usr/bin/env python
"""匿名の鍵の**登録簿**。作品フォルダーと頁に、公開してよい番号を1つずつ与える。

**公開物に作品名・頁のパスを出さない**(段 A)。だから公開表の行は番号で指す。
**番号は走行ごとに動いてはいけない** —— 動くと、**昨日の表の `W7` と今日の `W7` が
別の作品になる**。番号が動く表は、引用されたときに黙って別の物を指す。

だから番号は**登録簿に書き留めて読み戻す**(規律10 の形: 動く錨は先に登録する)。

| 鍵 | 指すもの | 登録簿 |
|---|---|---|
| `W<n>` | **作品フォルダー**(`<年の札>\\<作品>` の組) | `data/census_full/work_keys.csv` |
| `PN<n>` | **但し書きの付いた頁** | `data/census_full/page_keys.csv` |

**`W<n>` は墨の濃さの表が先に使っている**(`ink_darkness_work_ids.csv`、W1〜W12)。
**その12個を種にして、同じ番号空間を続ける** —— 同じ作品が2つの表に出たとき、
**同じ番号で指せないと突き合わせができない**。

**この器具が届かないもの**: 作品フォルダーが**同じ作品かどうか**(条例9: 同定は作者に問う)。
`W<n>` は**フォルダー1つに1つ**の番号で、**作品1つに1つではない** ——
同じ絵が2つの札に在れば番号は2つになる。
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK_KEYS = "data/census_full/work_keys.csv"
PAGE_KEYS = "data/census_full/page_keys.csv"
# **種**: 墨の濃さの表が先に配った番号。**この12個は動かさない**
INK_WORK_IDS = "data/census_full/ink_darkness_work_ids.csv"


def _read(path, key_col, val_col):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return {r[val_col]: r[key_col] for r in csv.DictReader(fh)}


def _write(path, key_col, val_col, mapping, extra):
    p = os.path.join(ROOT, path)
    rows = sorted(mapping.items(), key=lambda kv: _num(kv[1]))
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([key_col, val_col, "first_seen_in"])
        for val, key in rows:
            w.writerow([key, val, extra.get(val, "")])
    return len(rows)


def _num(key: str) -> int:
    return int("".join(c for c in key if c.isdigit()) or 0)


class Registry:
    """**開いて、引いて、閉じる。** 閉じるときに登録簿を書く。"""

    def __init__(self, path, prefix, key_col, val_col, seed=None):
        self.path, self.prefix = path, prefix
        self.key_col, self.val_col = key_col, val_col
        self.map = _read(path, key_col, val_col)
        self.n_loaded = len(self.map)
        self.n_seeded = 0
        if seed:
            for val, key in seed.items():
                if val not in self.map:
                    self.map[val] = key
                    self.n_seeded += 1
        self.first_seen = {}
        self.n_minted = 0

    def key_for(self, value: str, where: str = "") -> str:
        if value in self.map:
            return self.map[value]
        nxt = max((_num(k) for k in self.map.values()), default=0) + 1
        self.map[value] = f"{self.prefix}{nxt}"
        self.first_seen[value] = where
        self.n_minted += 1
        return self.map[value]

    def close(self) -> int:
        return _write(self.path, self.key_col, self.val_col,
                      self.map, self.first_seen)

    def report(self, label: str) -> None:
        print(f"    {label}: 登録簿から {self.n_loaded} / 種から {self.n_seeded}"
              f" / **新しく配った {self.n_minted}** / 合計 {len(self.map)}"
              "(**値は作品名・頁のパスなので印字しない**)")


def work_registry() -> Registry:
    """作品フォルダーの番号。**墨の濃さの W1〜W12 を種にする。**"""
    seed = {}
    p = os.path.join(ROOT, INK_WORK_IDS)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                seed[r["work_dir"]] = r["work_id"]
    return Registry(WORK_KEYS, "W", "work_id", "work_dir", seed=seed)


def page_registry() -> Registry:
    """但し書きの付いた頁の番号。**種は無い**(この表が最初)。"""
    return Registry(PAGE_KEYS, "PN", "page_key", "page_path")
