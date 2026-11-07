#!/usr/bin/env python3
"""聞き取りの UI を手元のサーバーで出す(指針 `doc/ui_guidelines_20261107.md`)。

使い方:
    ./venv/bin/python tools/ui/interview_server.py <questions.json> [--lan] [--port 8765]
        [--answers <out.jsonl>] [--image-root <dir>]

- 問いの形は `tools/ui/README.md`。画面は `tools/ui/interview_core.html`(artifact 版と共通)
- 答えは JSONL に **追記だけ**(上書きしない。直しは新しい行)。既定の置き場は問いの JSON の隣
  `<interview_id>_answers.jsonl` —— 原稿の名前を含むので、track の `data/` 配下に置くこと
- 既定の待ち受けは 127.0.0.1。**`--lan` で 0.0.0.0(LAN の端末から答えるため。2026-11-07 ユーザー決定)。
  インターネットには出さない** —— 起動時にそう印字する
- 絵は `--image-root`(既定: 問いの JSON のあるフォルダー)の中だけを出す。外へのパスは 403
"""
import argparse
import json
import socket
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from interview_spec import load_questions, render_page  # noqa: E402

TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".webp": "image/webp", ".gif": "image/gif", ".svg": "image/svg+xml"}


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.1.1", 9))  # 送らない。経路で選ばれる自分のアドレスを知るだけ
        return s.getsockname()[0]
    except OSError:
        return "?"
    finally:
        s.close()


def main():
    sys.stdout.reconfigure(line_buffering=True)  # nohup やログへ流したときも起動の印字が出るように
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--answers")
    ap.add_argument("--image-root")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--lan", action="store_true")
    a = ap.parse_args()

    qpath = Path(a.questions).resolve()
    spec = load_questions(qpath)  # 形の点検に落ちたらここで止まる
    root = Path(a.image_root).resolve() if a.image_root else qpath.parent
    out = Path(a.answers).resolve() if a.answers else qpath.parent / f"{spec['interview_id']}_answers.jsonl"
    lock = threading.Lock()

    # 絵の src は root からの相対パス → /img/<相対パス> で出す
    for it in spec["items"]:
        for im in it["images"]:
            p = (root / im["src"]).resolve()
            if root not in p.parents or not p.is_file():
                sys.exit(f"絵が無いか、image-root の外: {im['src']}")
            im["src"] = "img/" + im["src"]
    page = render_page(spec, mode="server").encode("utf-8")

    def read_rows():
        if not out.exists():
            return []
        with out.open(encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                return self.send(200, page, "text/html; charset=utf-8")
            if path == "/api/answers":
                with lock:
                    rows = read_rows()
                return self.send(200, json.dumps(rows, ensure_ascii=False).encode(), "application/json")
            if path.startswith("/img/"):
                p = (root / path[5:]).resolve()
                if root not in p.parents or not p.is_file():
                    return self.send(403, b"forbidden", "text/plain")
                return self.send(200, p.read_bytes(), TYPES.get(p.suffix.lower(), "application/octet-stream"))
            return self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/api/answer":
                return self.send(404, b"not found", "text/plain")
            try:
                row = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                assert row.get("interview_id") == spec["interview_id"]
                assert row.get("item_id") in spec["item_ids"]
            except Exception:
                return self.send(400, b"bad answer row", "text/plain")
            row["received_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            row["client"] = self.client_address[0]
            with lock, out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            return self.send(200, b"{}", "application/json")

    host = "0.0.0.0" if a.lan else "127.0.0.1"
    srv = ThreadingHTTPServer((host, a.port), H)
    n_prev = len(read_rows())
    print(f"聞き取り: {spec['interview_id']} / 問い {len(spec['items'])} / 既にある答えの行 {n_prev}")
    print(f"答えの置き場(追記だけ): {out}")
    if a.lan:
        print(f"**LAN に開いています**(0.0.0.0:{a.port})。LAN の端末から http://{lan_ip()}:{a.port}/ で開く。")
        print("**インターネットには出さないこと**(ルーターで外から入れない)。")
    else:
        print(f"手元だけで待っています: http://127.0.0.1:{a.port}/(LAN から答えるなら --lan)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"\n止めました。答えの行 {len(read_rows())}(この起動で +{len(read_rows()) - n_prev})")


if __name__ == "__main__":
    main()
