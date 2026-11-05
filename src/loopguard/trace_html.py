"""Self-contained static HTML trace viewer (single file, no external dependencies)."""

from __future__ import annotations

import html
import json
from typing import Any

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>loopguard trace __TITLE__</title>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e4df;
--run:#5b6cff;--step:#8a8f98;--model:#7c4dff;--tool:#0a9e7a;--error:#d64545;--blocked:#e08a00;--sel:#eef0ff}
@media (prefers-color-scheme:dark){:root{--bg:#111113;--panel:#1a1a1d;--fg:#ececec;--muted:#9a9aa3;
--line:#2c2c31;--sel:#24264a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:20px 24px 8px}h1{font-size:18px;margin:0 0 4px}
.sub{color:var(--muted);font-size:12px;font-family:ui-monospace,Menlo,monospace}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;padding:12px 24px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.card .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:20px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
.v.bad{color:var(--error)}.v.warn{color:var(--blocked)}.v.good{color:var(--tool)}
main{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);gap:12px;padding:0 24px 24px}
@media (max-width:860px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.row{display:grid;grid-template-columns:minmax(0,260px) 1fr 70px;align-items:center;gap:8px;
padding:5px 10px;border-bottom:1px solid var(--line);cursor:pointer}
.row:hover,.row.sel{background:var(--sel)}
.name{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:13px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
.track{position:relative;height:14px;background:transparent}
.bar{position:absolute;top:2px;height:10px;border-radius:3px;min-width:2px}
.dur{text-align:right;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.tag{font-size:10px;padding:1px 6px;border-radius:999px;margin-left:6px;color:#fff;vertical-align:middle}
#detail{padding:12px 14px}#detail h2{font-size:15px;margin:0 0 8px}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,Menlo,monospace;
background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;margin:6px 0}
.legend{display:flex;gap:14px;flex-wrap:wrap;padding:0 24px 10px;color:var(--muted);font-size:12px}
</style>
</head>
<body>
<header><h1>loopguard trace</h1><div class="sub" id="sub"></div></header>
<section class="cards" id="cards"></section>
<div class="legend" id="legend"></div>
<main><div class="panel" id="rows"></div><div class="panel" id="detail"><h2>Select a span</h2>
<div class="sub">Click any row to inspect attributes, tokens, cost and events. All values are PII-redacted.</div></div></main>
<script id="trace-data" type="application/json">__DATA__</script>
<script>
(function(){
const doc = JSON.parse(document.getElementById('trace-data').textContent);
const S = doc.summary, spans = doc.spans;
const colors = {run:'var(--run)',step:'var(--step)',model_call:'var(--model)',tool_call:'var(--tool)'};
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('sub').textContent = 'run ' + S.run_id;
const cards = [
  ['Status', S.status, S.status==='completed'?'good':(S.status==='incomplete'||S.status==='error'?'bad':'warn')],
  ['Duration', (S.duration_ms/1000).toFixed(2)+' s',''], ['Model calls', S.model_calls,''],
  ['Tool calls', S.tool_calls,''], ['Tokens', S.tokens.toLocaleString(),''],
  ['Cost', '$'+Number(S.cost_usd).toFixed(4),''], ['Retries', S.retries, S.retries?'warn':''],
  ['Blocked', S.blocked_actions, S.blocked_actions?'warn':''],
  ['Injections', S.injections_flagged, S.injections_flagged?'bad':''], ['Errors', S.errors, S.errors?'bad':'']];
document.getElementById('cards').innerHTML = cards.map(([k,v,c]) =>
  `<div class="card"><div class="k">${k}</div><div class="v ${c}">${esc(v)}</div></div>`).join('');
document.getElementById('legend').innerHTML = Object.entries(colors).map(([k,c]) =>
  `<span><span class="dot" style="background:${c}"></span>${k}</span>`).join('') +
  '<span><span class="dot" style="background:var(--blocked)"></span>blocked</span>' +
  '<span><span class="dot" style="background:var(--error)"></span>error</span>';
const byId = {}; spans.forEach(s => { byId[s.span_id] = s; s.children = []; });
const roots = [];
spans.forEach(s => (s.parent_id && byId[s.parent_id] ? byId[s.parent_id].children.push(s) : roots.push(s)));
const t0 = Math.min(...spans.map(s => s.start)), t1 = Math.max(...spans.map(s => s.end || s.start));
const total = Math.max(t1 - t0, 1e-6);
const ordered = [];
(function walk(list, depth){ list.sort((a,b)=>a.start-b.start).forEach(s => { s.depth = depth; ordered.push(s); walk(s.children, depth+1); }); })(roots, 0);
const rowsEl = document.getElementById('rows');
rowsEl.innerHTML = ordered.map((s,i) => {
  const color = s.status==='error' ? 'var(--error)' : s.status==='blocked' ? 'var(--blocked)' : (colors[s.kind]||'var(--step)');
  const left = ((s.start - t0)/total*100).toFixed(3), width = (((s.end||s.start) - s.start)/total*100).toFixed(3);
  let tags = '';
  if (s.status==='blocked') tags += '<span class="tag" style="background:var(--blocked)">blocked</span>';
  if (s.status==='error') tags += '<span class="tag" style="background:var(--error)">error</span>';
  if (s.attributes && s.attributes.injection_flagged) tags += '<span class="tag" style="background:var(--error)">injection</span>';
  if ((s.events||[]).some(e => e.name==='retry')) tags += '<span class="tag" style="background:var(--model)">retry</span>';
  return `<div class="row" data-i="${i}"><div class="name" style="padding-left:${s.depth*14}px">` +
    `<span class="dot" style="background:${color}"></span>${esc(s.name)}${tags}</div>` +
    `<div class="track"><div class="bar" style="left:${left}%;width:${width}%;background:${color}"></div></div>` +
    `<div class="dur">${s.duration_ms.toFixed(1)} ms</div></div>`;
}).join('');
rowsEl.addEventListener('click', ev => {
  const row = ev.target.closest('.row'); if (!row) return;
  rowsEl.querySelectorAll('.row.sel').forEach(r => r.classList.remove('sel')); row.classList.add('sel');
  const s = ordered[+row.dataset.i];
  document.getElementById('detail').innerHTML =
    `<h2>${esc(s.name)}</h2><div class="sub">${esc(s.kind)} · ${esc(s.status)} · ${s.duration_ms.toFixed(1)} ms</div>` +
    `<h3 style="font-size:13px;margin:12px 0 0">Attributes</h3><pre>${esc(JSON.stringify(s.attributes, null, 2))}</pre>` +
    ((s.events||[]).length ? `<h3 style="font-size:13px;margin:12px 0 0">Events</h3><pre>${esc(JSON.stringify(s.events, null, 2))}</pre>` : '');
});
})();
</script>
</body>
</html>
"""


def render_html(doc: dict[str, Any]) -> str:
    data = json.dumps(doc, default=str)
    # Make the JSON safe to embed inside a <script> element.
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    title = html.escape(str(doc.get("trace_id", "")))
    return _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)
