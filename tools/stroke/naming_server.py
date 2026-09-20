#!/usr/bin/env python
"""Naming web UI for word semantics (plan gate 5), localhost only.

Avoids the error-prone CSV column mapping: the UI keys every name directly to the
word id shown on the image. Per word it shows 8 truth examples + the word-only
prototype, plus the cluster highlighted in panel context. Names autosave to
results/word_semantics_20260920/naming_ui.json via POST.

Run:  python tools/stroke/naming_server.py        (then open http://localhost:8471)
"""
import csv, json, sys
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook import load
from train_codebook2 import Codebook2, topn_mask
from codebook_diag import draw

CLUSTERS = "results/cluster_set_20260919"
PACK = "results/panel_pack_20260919"
CKPT = "results/invariance_20260920/l4_w500/codebook2.pt"
OUT = Path("results/word_semantics_20260920/ui")
SAVE = Path("results/word_semantics_20260920/naming_ui.json")
PT = Path("results/panel_composition_20260921")   # panel-type naming
PT_SAVE = PT / "type_names.json"
PORT = 8471
N_TOP, N_EX, SIZE, H_CANVAS = 20, 8, 150, 220


def build_images():
    dev = torch.device("cuda")
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    m = Codebook2(levels=tuple(ck["levels"]), coord_bins=ck["args"].get("coord_bins", 0)).to(dev)
    m.load_state_dict(ck["model"]); m.eval(); m.rfsq.stages = ck["args"].get("stages", 3)
    Pt, Wt, Mt, mt = load(CLUSTERS, "train", dev)
    Pv, Wv, Mv, mv = load(CLUSTERS, "test", dev)
    P = torch.cat([Pt, Pv]); W = torch.cat([Wt, Wv]); M = torch.cat([Mt, Mv]); mrows = mt + mv
    words = []
    with torch.no_grad():
        for i in range(0, len(P), 4096):
            _q, w = m.encode(P[i:i + 4096].float(), W[i:i + 4096].float(), M[i:i + 4096], 1)
            words.append(w.cpu())
    words = torch.cat(words).numpy()

    used = np.unique(words)
    levels = tuple(ck["levels"])
    basis = np.cumprod((1,) + levels[:-1])
    half_w = np.floor(np.array(levels) / 2)
    q = np.zeros((len(used), len(levels)), np.float32)
    for i, w in enumerate(used):
        d = (int(w) // basis) % np.array(levels)
        q[i] = (d - half_w) / half_w
    with torch.no_grad():
        ex, pp, pw, cl, _lg = m.decode(torch.from_numpy(q).to(dev))
        keep = topn_mask(ex.float(), cl.float())
    proto = {int(w): (pp[i].float().cpu().numpy(), keep[i].cpu().numpy()) for i, w in enumerate(used)}

    tr_mask = np.array([r["split"] == "train" for r in mrows])
    freq = np.bincount(words[tr_mask], minlength=500)
    top = [int(w) for w in np.argsort(-freq)[:N_TOP]]

    panels_csv = list(csv.DictReader(open(f"{PACK}/panels.csv")))
    strokes = np.load(f"{PACK}/strokes.npy", mmap_mode="r")

    def render_ctx(panel_id, cl_pts, cl_mask, cy, cx, scale):
        pr = panels_csv[panel_id]
        rows = np.asarray(strokes[int(pr["start"]):int(pr["start"]) + int(pr["n"])])
        ppn = rows[:, :32].reshape(-1, 16, 2)
        ink = np.abs(ppn).sum(-1) > 0
        H = max(int(np.ceil(ppn[..., 1][ink].max())) + 2, 2) if ink.any() else 2
        Wd = max(int(np.ceil(ppn[..., 0][ink].max())) + 2, 2) if ink.any() else 2
        z = H_CANVAS / H
        canvas = np.full((H_CANVAS, max(int(Wd * z), 2), 3), 255, np.uint8)

        def xf(pts2):
            return np.round(pts2 * z)[:, ::-1].astype(np.int32).reshape(-1, 1, 2)

        for s in range(ppn.shape[0]):
            if not ink[s].any():
                continue
            xy = np.clip(xf(ppn[s][ink[s]]), 0, [canvas.shape[1] - 1, H_CANVAS - 1])
            cv2.polylines(canvas, [xy], False, (205, 205, 205), 1)
        nat = (cl_pts / scale) + torch.tensor([cy, cx]).to(cl_pts.device)
        for s in range(cl_pts.shape[0]):
            if not cl_mask[s]:
                continue
            xy = np.clip(xf(nat[s].cpu().numpy()), 0, [canvas.shape[1] - 1, H_CANVAS - 1])
            cv2.polylines(canvas, [xy], False, (30, 30, 230), 2)
        return canvas

    rng = np.random.default_rng(0)
    meta = []
    for w in top:
        exs = [j for j in np.flatnonzero(words == w) if tr_mask[j]][:N_EX]

        def tag_img(img, text):
            cv2.putText(img, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 2)
            return img

        cells = [tag_img(draw(P[j].cpu().numpy(), M[j].cpu().numpy(), SIZE), f"ex{k + 1}")
                 for k, j in enumerate(exs)]
        while len(cells) < N_EX:
            cells.append(np.full((SIZE, SIZE, 3), 255, np.uint8))
        pr, pk = proto[w]
        cells.append(tag_img(draw(pr, pk, SIZE), "proto"))
        cv2.imwrite(str(OUT / f"word_{w}_ex.png"), np.hstack(cells))
        pool = [j for j in np.flatnonzero(words == w) if tr_mask[j]]
        pick = rng.choice(pool, min(4, len(pool)), replace=False) if pool else []
        ctx = []
        for j in pick:
            r = mrows[int(j)]
            ctx.append(render_ctx(int(r["panel"]), P[int(j)], M[int(j)],
                                  float(r["cy"]), float(r["cx"]), float(r["scale"])))
        if ctx:
            wmax = max(c.shape[1] for c in ctx)
            ctx = [np.pad(c, ((0, 0), (0, wmax - c.shape[1]), (0, 0)), constant_values=255) for c in ctx]
            cv2.imwrite(str(OUT / f"word_{w}_ctx.png"), np.hstack(ctx))
        meta.append({"word": w, "freq": int(freq[w])})
    (OUT / "words.json").write_text(json.dumps(meta))
    return meta


HTML = """<!doctype html><meta charset="utf-8"><title>word naming</title>
<style>
 body{font-family:sans-serif;margin:16px} img{image-rendering:auto;border:1px solid #ccc}
 #ex img{width:147px} #ctx img{height:200px;margin-right:6px}
 input{font-size:20px;width:400px} button{font-size:16px}
 #list span{display:inline-block;margin:2px;padding:2px 6px;border:1px solid #999;cursor:pointer}
 .named{background:#cfc} .cur{outline:3px solid #e33}
</style>
<body>
<h2>word <span id="wid"></span> — freq <span id="fr"></span> (<span id="done"></span>)</h2>
<div id="ex"></div><div id="ctx"></div>
<p>名前: <input id="name"> <button id="prev">← prev</button> <button id="next">next →</button>
<span id="saved"></span></p>
<p>並び: 上段 左から8例=真値の出現例、最右=単語のみ原型。下段=コマ内での位置(灰=コマ全体、赤=この単語)。
名前は単語IDに直接保存される(CSVの列対応は無い)。← → キーで移動。</p>
<div id="list"></div>
<script>
let words = [], cur = 0, names = {};
async function boot(){
  words = await (await fetch('ui/words.json')).json();
  try{ names = await (await fetch('ui/naming_ui.json')).json(); }catch(e){}
  render_list(); show(0);
}
function render_list(){
  document.getElementById('list').innerHTML = words.map((w,i)=>
    `<span class="${names[w.word]?'named':''} ${i===cur?'cur':''}" onclick="show(${i})">${w.word}</span>`).join('');
  document.getElementById('done').textContent =
    Object.keys(names).filter(k=>names[k]).length + '/' + words.length + ' named';
}
function show(i){
  cur = i; const w = words[i];
  document.getElementById('wid').textContent = w.word;
  document.getElementById('fr').textContent = w.freq;
  document.getElementById('ex').innerHTML = `<img src="ui/word_${w.word}_ex.png">`;
  document.getElementById('ctx').innerHTML = `<img src="ui/word_${w.word}_ctx.png">`;
  document.getElementById('name').value = names[w.word] || '';
  render_list();
}
async function save(v){
  const w = words[cur].word;
  if(v) names[w] = v; else delete names[w];
  await fetch('/save', {method:'POST', body: JSON.stringify({word: w, name: v})});
  const s = document.getElementById('saved'); s.textContent = 'saved';
  setTimeout(()=>s.textContent='', 800); render_list();
}
document.getElementById('name').addEventListener('change', e=>save(e.target.value.trim()));
document.getElementById('name').addEventListener('keydown', e=>{
  if(e.key==='Enter'){ e.preventDefault(); show(Math.min(cur+1, words.length-1)); document.getElementById('name').focus(); }});
document.getElementById('prev').onclick = ()=>show(Math.max(cur-1,0));
document.getElementById('next').onclick = ()=>show(Math.min(cur+1, words.length-1));
document.addEventListener('keydown', e=>{
  if(document.activeElement.tagName==='INPUT') return;
  if(e.key==='ArrowLeft') show(Math.max(cur-1,0));
  if(e.key==='ArrowRight') show(Math.min(cur+1, words.length-1));
});
boot();
</script>"""


HTML_TYPES = """<!doctype html><meta charset="utf-8"><title>panel type naming</title>
<style>
 body{font-family:sans-serif;margin:16px}
 .row{margin:14px 0;padding:8px;border:1px solid #ccc}
 .row img{width:100%;max-width:1200px;border:1px solid #999;display:block}
 .info{color:#555;font-size:13px;margin:4px 0}
 input{font-size:18px;width:480px}
 .named{background:#cfc}
 #done{font-size:18px}
</style>
<body>
<h2>コマ型の命名 (k=12、行=サイズ順) <span id="done"></span></h2>
<p>各ブロックの上が代表例8枚(中心に近い順)、下がタグ所見。名前を入れると即保存。
複数語は "/" 区切り。<a href="/">単語命名はこちら</a></p>
<div id="rows"></div>
<script>
let types = [], names = {};
async function boot(){
  types = await (await fetch('pt/types.json')).json();
  try{ names = await (await fetch('pt/names.json')).json(); }catch(e){}
  document.getElementById('rows').innerHTML = types.map(t=>`
    <div class="row" id="row${t.rank}">
      <b>#${t.rank + 1}</b> (cluster ${t.cluster}, n=${t.n}, ${t.pct}%)
      <input id="in${t.rank}" value="${names[t.rank] || ''}"
             onchange="save(${t.rank}, this.value.trim())">
      <div class="info">${t.median}</div>
      <img src="pt/${t.img}">
      <div class="info">tags: ${t.tags}</div>
    </div>`).join('');
  update();
}
async function save(rank, v){
  if(v) names[rank] = v; else delete names[rank];
  await fetch('/save_type', {method:'POST', body: JSON.stringify({rank: rank, name: v})});
  update();
}
function update(){
  document.getElementById('done').textContent =
    Object.keys(names).filter(k=>names[k]).length + '/' + types.length + ' named';
}
boot();
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/" or p == "/index.html":
            self._send(HTML.encode(), "text/html; charset=utf-8")
        elif p == "/types":
            self._send(HTML_TYPES.encode(), "text/html; charset=utf-8")
        elif p == "/pt/types.json":
            self._send((PT / "types.json").read_bytes(), "application/json")
        elif p == "/pt/names.json":
            data = PT_SAVE.read_bytes() if PT_SAVE.exists() else b"{}"
            self._send(data, "application/json")
        elif p.startswith("/pt/rows/") and ".." not in p:
            f = PT / p[len("/pt/"):]
            if f.exists():
                self._send(f.read_bytes(), "image/png")
            else:
                self.send_error(404)
        elif p == "/ui/naming_ui.json":
            data = SAVE.read_bytes() if SAVE.exists() else b"{}"
            self._send(data, "application/json")
        elif p.startswith("/ui/") and ".." not in p:
            f = OUT / p[len("/ui/"):]
            if f.exists():
                self._send(f.read_bytes(), "image/png" if f.suffix == ".png" else "application/json")
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path == "/save_type":
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n))
            names = json.loads(PT_SAVE.read_text()) if PT_SAVE.exists() else {}
            if d.get("name"):
                names[str(d["rank"])] = d["name"]
            else:
                names.pop(str(d["rank"]), None)
            PT_SAVE.write_text(json.dumps(names, ensure_ascii=False, indent=1))
            self._send(b"ok", "text/plain")
        elif urlparse(self.path).path == "/save":
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n))
            names = json.loads(SAVE.read_text()) if SAVE.exists() else {}
            if d.get("name"):
                names[str(d["word"])] = d["name"]
            else:
                names.pop(str(d["word"]), None)
            SAVE.write_text(json.dumps(names, ensure_ascii=False, indent=1))
            self._send(b"ok", "text/plain")
        else:
            self.send_error(404)

    def _send(self, b, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    meta = build_images()
    print(f"{len(meta)} words -> {OUT}; naming -> {SAVE}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
