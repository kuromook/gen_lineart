"""M50: **L のサーバを書き換えずに、保存先を M の木へ向ける起動器**。

2026-10-26、行列の鍵を直して 8767 を立て直したとき、**起動の門の印字が
2つめと3つめの副作用を見せた**:

```
異常の報告: ... -> /home/sh1/deepl/lineart-fill-region/labels/fill_anomaly.jsonl  既報 15 件
go() 実行 [order=15] ... info: ...  ⚑ 異常として報告済み: 髪ベタ塗りつぶしが背景の木に…
```

| | 副作用 | 原因 |
|---|---|---|
| **(1)** | 異常の報告が **L の木**に書かれる(前便で記録済み) | `S['flagfile'] = LAB / 'fill_anomaly.jsonl'` が**引数になっていない** |
| **(2)** | **L の既報が M の問に出る**(`order` 11 と 15 が衝突) | 異常は **`order` で引かれる**。別の行列でも同じ番号なら出る |
| **(3)** | **`--save labels/...` が L の木に落ちる** | `sv = ROOT / a.save`。**`ROOT` はスクリプトの位置から決まる** = L の木 |

**(2) は作者に見える汚染である** —— **別の問いに付いた異常の注記を、M の問の上で読むことになる。**
**(3) は答が消える形ではないが、置き場が違う**(L の `labels/area_count.jsonl` に落ちていた。
まだ1件も保存されていないので失った答は無い)。

**L のサーバは触らない**(L は busy / 器具の持ち主は L)。
**import して、モジュールの定数だけを M の木に向け直す。**
**向け直したことを門で確かめてから起動する** —— 向け直し損ねたら、**黙って L の木に書く**。

```
./venv/bin/python tools/ui/m50_launch_area_count.py \
    --host 0.0.0.0 --port 8767 --queue results/m47_ui/queue.json \
    --save labels/area_count.jsonl --extra ''
```
"""
import sys
from pathlib import Path

MTREE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(MTREE.parent / 'lineart-fill-region/tools/ui'))
import l19_fill_label_server as l19                       # noqa: E402

for name in ('ROOT', 'LAB', 'load_state', 'main', 'S'):
    if not hasattr(l19, name):
        raise SystemExit(f'**L のサーバに `{name}` が無い。起動しない** '
                         f'(流用の接点が変わった。m50 を読み直すこと)')

l19.ROOT = MTREE                      # `--save` の相対を M の木で解決させる
l19.LAB = MTREE / 'labels'            # 異常の報告も M の木へ
l19.LAB.mkdir(exist_ok=True)

_orig = l19.load_state


def load_state(save_path, *a, **k):
    """**門**: 読み書きする2つのファイルが M の木の下に在ることを確かめる。"""
    _orig(save_path, *a, **k)
    for key, what in (('save', '答の保存先'), ('flagfile', '異常の報告')):
        p = Path(l19.S[key]).resolve()
        if MTREE not in p.parents:
            raise SystemExit(f'**{what} が M の木の外: {p}。起動しない**')
        print(f'  {what}: {p}', flush=True)
    print('  **門: 保存先も異常の報告も M の木の下**(L の木には書かない)',
          flush=True)


l19.load_state = load_state
l19.main()
