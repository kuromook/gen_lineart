// **解答欄が本当に出るかを、go() を走らせて確かめる。**
//
// 2026-10-26 (基盤の報告): 行列を流用したとき `area` が `area_px` という名で入っており、
// **解答欄のボタンが1つも出ず、ショートカットキーも全部死んだ**まま作者に渡った。
// 渡す前に見ていたのは「起動の門を通った」と「HTTP 200」だけで、**どちらも go() を
// 走らせていない。** 構文検査(門その2)も通る種類の欠陥だった。
//
// 使い方: node l19_ui_smoke.js <payloads.json> <ui.js>
//   payloads.json = {ラベル: /api/item が返す記録}  ui.js = PAGE の <script> の中身
// 1つでも解答欄が空なら終了コード 1。
const fs = require('fs'), vm = require('vm');
const payloads = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const src = fs.readFileSync(process.argv[3], 'utf8');

function mkEl() {
  const e = {children: [], style: {}, appendChild(c) { e.children.push(c); }};
  Object.defineProperty(e, 'innerHTML',
    {set(v) { e._h = v; if (v === '') e.children = []; }, get() { return e._h || ''; }});
  Object.defineProperty(e, 'textContent', {set(v) { e._t = v; }, get() { return e._t || ''; }});
  Object.defineProperty(e, 'onclick', {set(f) { e._c = f; }, get() { return e._c; }});
  Object.defineProperty(e, 'className', {set(v) { e._n = v; }, get() { return e._n || ''; }});
  return e;
}

(async () => {
  let bad = 0;
  for (const lab of Object.keys(payloads)) {
    const it = payloads[lab], els = {};
    const sb = {console, JSON, Math, Date, setTimeout, performance: {now: () => 0},
      document: {getElementById: k => (els[k] = els[k] || mkEl()),
                 createElement: () => mkEl(), addEventListener: () => {}},
      window: {}, addEventListener: () => {}, location: {href: ''},
      prompt: () => null, confirm: () => false, alert: () => {},
      fetch: async () => ({ok: true, status: 200, json: async () => it})};
    sb.globalThis = sb;
    const ctx = vm.createContext(sb);
    let err = null;
    try { vm.runInContext(src, ctx); await vm.runInContext('go(0)', ctx); }
    catch (e) { err = e; }
    const nw = (els['words'] || {children: []}).children.length;
    const no = (els['other'] || {children: []}).children.length;
    const km = err ? [] : vm.runInContext('Object.keys(keymap)', ctx);
    const ok = err ? [] : vm.runInContext('Object.keys(okeys)', ctx);
    console.log(`  go() 実行 [${lab}]  語ボタン ${nw}  その他 ${no}  ` +
                `keymap ${km.length}  okeys ${ok.length}` +
                (err ? `  **例外 ${err.constructor.name}: ${err.message}**` : ''));
    console.log(`    info: ${(els['info'] || {}).textContent || '(空)'}`);
    // **語が1つも無い行列もあり得る**(語を足さない使い方)。その他ボタンは必ず出る
    if (err || no === 0 || ok.length === 0 || (it.words || []).length !== nw) {
      bad++;
      console.log('    **解答欄が出ていない**');
    }
  }
  process.exit(bad ? 1 : 0);
})();
