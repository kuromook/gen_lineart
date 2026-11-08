#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L38 **分解の台** —— 作者が面を**選択ツールで区分けして名を付ける**画面(2026-11-08)。

作者の設計(逐語):
> **別途ＵＩで、分解しなければならない面をユーザ操作の選択ツールで区分けして名をつける**

**これは聞き取りではなく裁定を集める器具である**(規律6。点数ではない)。
だから基盤の共通UI(選択肢を1つ選ぶ形)には乗らない。**L の中に置き、一般に使えると分かったら
基盤に出す**(共通UIは基盤のもの)。

- **絵は原寸 1:1**(規律6)。指で拡大・移動できる。**縮小モンタージュで論じない**
- **機械の提案を先に見せない**(登録 §6)—— 見せたら裁定が提案に寄り、提案を採点できなくなる。
  **この画面に腕Wの出力は一切出さない**
- 答えは JSONL に **追記だけ**(上書きしない。直しは新しい行)。**区分けの地図は PNG で別に置く**
- 既定の待ち受けは 127.0.0.1。**`--lan` で 0.0.0.0。インターネットには出さない**
- **原稿の画素を含むものは `data/` 配下**(`.gitignore` 済み。絶対にコミットしない)

    .../python tools/ui/l38_decompose_bench_20261108.py \
        --units results/decompose20261108/windows.json --lan --port 8766
"""
import argparse
import base64
import json
import re
import socket
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
PAGE = HERE / 'l38_decompose_bench.html'
FID = re.compile(r'^f[0-9]{2,4}$')


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('192.168.1.1', 9))
        return s.getsockname()[0]
    except OSError:
        return '?'
    finally:
        s.close()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--units', required=True, help='`windows.json`(切り出しの器具が書いたもの)')
    ap.add_argument('--crops', help='切り出しの置き場(既定: units と同じフォルダー)')
    ap.add_argument('--out', help='裁定の置き場(JSONL。追記だけ)')
    ap.add_argument('--labels', help='区分けの地図(PNG)の置き場')
    ap.add_argument('--only-split', help='チェックの答え(JSONL)。**「分解しないといけない」と'
                    '答えた単位だけを台に乗せる**。落とした数を理由別に印字する(規律9)')
    ap.add_argument('--port', type=int, default=8766)
    ap.add_argument('--lan', action='store_true')
    a = ap.parse_args()

    units = json.load(open(a.units, encoding='utf-8'))
    crops = Path(a.crops).resolve() if a.crops else Path(a.units).resolve().parent
    out = Path(a.out).resolve() if a.out else crops / 'l38_decompose_rulings.jsonl'
    labels = Path(a.labels).resolve() if a.labels else out.parent / 'labels'
    labels.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    miss = [u['fid'] for u in units
            if not (crops / f'{u["fid"]}.png').is_file()
            or not (crops / f'{u["fid"]}_mask.png').is_file()]
    if miss:
        sys.exit(f'**絵かマスクが無い単位 {len(miss)}: {miss[:8]}** —— 揃うまで出さない')
    # **画面に渡すのは fid だけ。素性も群の札も渡さない**(作者の裁定を機械の見立てに寄せない)
    ids = [u['fid'] for u in units]
    if a.only_split:
        last = {}
        with open(a.only_split, encoding='utf-8') as f:
            for x in f:
                if x.strip():
                    r = json.loads(x)
                    last[r.get('item_id')] = r.get('value')   # **最後の行が有効。前の行は消さない**
        drop = {}
        keep = []
        for fid in ids:
            v = last.get(fid)
            if v == 'split':
                keep.append(fid)
            else:
                drop[v or '(まだ答えが無い)'] = drop.get(v or '(まだ答えが無い)', 0) + 1
        print(f'チェックの答えで絞った: 台に乗せる {len(keep)} / 元 {len(ids)}')
        for k, v in sorted(drop.items(), key=lambda t: -t[1]):
            print(f'  落とした {v:>3}  答えが {k}')
        ids = keep
        if not ids:
            sys.exit('**「分解しないといけない」と答えた単位が1つも無い。台を出す相手がいない**')
    page = PAGE.read_text(encoding='utf-8').replace(
        '/*__UNITS__*/null', json.dumps(ids, ensure_ascii=False)).encode('utf-8')
    lock = threading.Lock()

    def rows():
        if not out.exists():
            return []
        with out.open(encoding='utf-8') as f:
            return [json.loads(x) for x in f if x.strip()]

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ('/', '/index.html'):
                return self.send(200, page, 'text/html; charset=utf-8')
            if path == '/api/done':
                with lock:
                    seen = {}
                    for r in rows():          # **最後の行が有効。前の行は消さない**
                        seen[r.get('item_id')] = r.get('value')
                return self.send(200, json.dumps(seen, ensure_ascii=False).encode(),
                                 'application/json')
            m = re.fullmatch(r'/(img|mask)/(f[0-9]{2,4})\.png', path)
            if m:
                p = crops / (m.group(2) + ('_mask.png' if m.group(1) == 'mask' else '.png'))
                if not p.is_file():
                    return self.send(404, b'not found', 'text/plain')
                return self.send(200, p.read_bytes(), 'image/png')
            return self.send(404, b'not found', 'text/plain')

        def do_POST(self):
            if urlparse(self.path).path != '/api/partition':
                return self.send(404, b'not found', 'text/plain')
            try:
                row = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                assert FID.fullmatch(row.get('item_id') or '')
                assert row.get('item_id') in ids
                assert row.get('value') in ('split', 'whole', '__unknown__')
            except Exception:
                return self.send(400, b'bad ruling row', 'text/plain')
            png = row.pop('label_png', None)
            row['received_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
            row['client'] = self.client_address[0]
            with lock:
                if png and png.startswith('data:image/png;base64,'):
                    # **版を上書きしない。時刻を名前に入れて増やす**(規律7)
                    stamp = row['received_at'].replace(':', '').replace('-', '')[:15]
                    f = labels / f'{row["item_id"]}_{stamp}.png'
                    f.write_bytes(base64.b64decode(png.split(',', 1)[1]))
                    row['label_file'] = f.name
                with out.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                    f.flush()
            return self.send(200, b'{}', 'application/json')

    host = '0.0.0.0' if a.lan else '127.0.0.1'
    srv = ThreadingHTTPServer((host, a.port), H)
    n0 = len(rows())
    print(f'分解の台: 単位 {len(ids)} / 既にある裁定の行 {n0}')
    print(f'裁定の置き場(追記だけ): {out}')
    print(f'区分けの地図: {labels}/<単位>_<時刻>.png(**上書きしない**)')
    if a.lan:
        print(f'**LAN に開いています**(0.0.0.0:{a.port})。http://{lan_ip()}:{a.port}/ で開く。')
        print('**インターネットには出さないこと**。')
    else:
        print(f'手元だけ: http://127.0.0.1:{a.port}/(LAN から使うなら --lan)')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f'\n止めました。裁定の行 {len(rows())}(この起動で +{len(rows()) - n0})')


if __name__ == '__main__':
    main()
