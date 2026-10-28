// **選べるかを、画面の JS を実際に走らせて確かめる。**
//
// 2026-10-28: 作者から「**選択ができない どういう状況かも不明**」。
// それまでに通っていたのは「起動の門」「HTTP 200」「寸法の一致」だけで、
// **どれも画面の JS を1行も走らせていない**。L19 が 2026-10-26 に同じ形で踏んでいる
// (`l19_ui_smoke.js` の冒頭)。**同じ罠を、同じ器具の族で2度踏んだ。**
//
// 使い方: node l38_ui_smoke.js <payload.json> <ui.js>
//   payload.json = /state が返す記録   ui.js = 画面の <script> の中身
// 面の札・一覧が出ない、選択しても状態が変わらない、draw が落ちる → 終了コード 1。
const fs = require('fs'), vm = require('vm');
const it = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const src = fs.readFileSync(process.argv[3], 'utf8');

// **画像が縮んで描画されている状況を作る。** 窓が狭いとき flex は画像を縮めるので、
// 「縮んでいない前提」で座標を出す実装はここで落ちる(2026-10-28 の「選択ができない」)。
const SHRINK = 0.5;
function mkEl(w, h) {
  const e = {children: [], style: {}, appendChild(c) { e.children.push(c); },
             getBoundingClientRect: () => ({left: 0, top: 0,
                                            width: (w || 100) * SHRINK,
                                            height: (h || 100) * SHRINK})};
  Object.defineProperty(e, 'innerHTML',
    {set(v) { e._h = v; if (v === '') e.children = []; }, get() { return e._h || ''; }});
  Object.defineProperty(e, 'textContent', {set(v) { e._t = v; }, get() { return e._t || ''; }});
  Object.defineProperty(e, 'onclick', {set(f) { e._c = f; }, get() { return e._c; }});
  Object.defineProperty(e, 'className', {set(v) { e._n = v; }, get() { return e._n || ''; }});
  Object.defineProperty(e, 'title', {set(v) { e._ti = v; }, get() { return e._ti || ''; }});
  Object.defineProperty(e, 'src', {set(v) { e._s = v; }, get() { return e._s || ''; }});
  Object.defineProperty(e, 'width', {set(v) { e._w = v; }, get() { return e._w; }});
  return e;
}

let REJ = null;
const HITS = [];
process.on('unhandledRejection', e => { REJ = REJ || e; });

(async () => {
  const els = {}, posted = [];
  const sb = {console, JSON, Math, Date, setTimeout, Set, Array, Object, String, Number,
    document: {getElementById: k => (els[k] = els[k] ||
                 (k === 'im' ? mkEl(it.sel_w, it.sel_h) : mkEl())),
               createElement: () => mkEl(), addEventListener: () => {}},
    window: {}, addEventListener: () => {}, location: {href: ''},
    encodeURIComponent: encodeURIComponent,
    prompt: () => '試験の語', confirm: () => true, alert: m => posted.push('alert:' + m),
    fetch: async (u, o) => {
      if (o && o.method === 'POST') { posted.push(JSON.parse(o.body)); }
      if (u.indexOf('/hit') >= 0) { HITS.push(u); }
      return {ok: true, status: 200,
              json: async () => (u.indexOf('/hit') >= 0 ? {i: 1}
                                 : o && o.method === 'POST' ? {next: it.order} : it)};
    }};
  sb.globalThis = sb;
  const ctx = vm.createContext(sb);
  let err = null;
  try {
    vm.runInContext(src, ctx);
    await new Promise(r => setTimeout(r, 30));          // load(0) の fetch を待つ
  } catch (e) { err = e; }
  const nb = (els['badges'] || {children: []}).children.length;
  const nl = (els['list'] || {children: []}).children.length;
  const nf = it.faces.length;
  let selTxt0 = (els['sel'] || {}).textContent || '';
  // ---- 札を押す(= 面を選ぶ)
  let tErr = null, selTxt1 = '', onCnt = 0;
  try {
    const b = (els['badges'].children[0] || {});
    if (b._c) b._c({stopPropagation() {}});
    await new Promise(r => setTimeout(r, 10));
    selTxt1 = els['sel'].textContent || '';
    onCnt = els['badges'].children.filter(c => (c.className || '').indexOf('on') >= 0).length;
  } catch (e) { tErr = e; }
  // ---- 一覧の行を押す
  let lErr = null, selTxt2 = '';
  try {
    const r2 = (els['list'].children[1] || els['list'].children[0] || {});
    if (r2._c) r2._c();
    await new Promise(r => setTimeout(r, 10));
    selTxt2 = els['sel'].textContent || '';
  } catch (e) { lErr = e; }
  // ---- **縮んだ画像**の上で、面1の代表点に当たる位置を押す
  let hErr = null, hitOk = null;
  try {
    const f = it.faces[0];
    const cx = f.sx / it.sel_w * (it.sel_w * SHRINK);   // 描画された座標
    const cy = f.sy / it.sel_h * (it.sel_h * SHRINK);
    await ctx.hit({clientX: cx, clientY: cy});
    await new Promise(r => setTimeout(r, 10));
    const q = HITS[HITS.length - 1] || '';
    const gy = +((q.match(/[?&]y=(-?\d+)/) || [])[1]);
    const gx = +((q.match(/[?&]x=(-?\d+)/) || [])[1]);
    const wy = Math.round(f.sy / it.scale), wx = Math.round(f.sx / it.scale);
    hitOk = {送った: [gy, gx], 期待: [wy, wx],
             ずれ: [Math.abs(gy - wy), Math.abs(gx - wx)]};
  } catch (e) { hErr = e; }

  const out = {err: err && String(err), 握り潰された誤り: REJ && String(REJ.stack || REJ), 面: nf, 札: nb, 一覧: nl,
               選択前: selTxt0, 札を押した後: selTxt1, 選択中の札: onCnt,
               一覧も押した後: selTxt2,
               札の誤り: tErr && String(tErr), 一覧の誤り: lErr && String(lErr),
               縮んだ画像を押す: hitOk, 縮めた倍率: SHRINK,
               押した誤り: hErr && String(hErr)};
  console.log(JSON.stringify(out, null, 1));
  const TOL = 3;   // 丸めの許容(画素)
  const bad = err || REJ || tErr || lErr || hErr || nb !== nf || nl !== nf ||
              selTxt1 === selTxt0 || onCnt !== 1 || selTxt2 === selTxt1 ||
              !hitOk || hitOk.ずれ[0] > TOL || hitOk.ずれ[1] > TOL;
  console.log(bad ? '**選べない。止める**' : '選べる(札・一覧の両方)');
  process.exit(bad ? 1 : 0);
})();
