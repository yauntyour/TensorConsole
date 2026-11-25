// UI primitives: dom, formatting, toasts, modals, tables
const UI = {
  el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') n.className = v;
      else if (k === 'html') n.innerHTML = v;
      else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
      else if (k === 'style' && typeof v === 'object') Object.assign(n.style, v);
      else if (v !== null && v !== undefined && v !== false) n.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat(9)) {
      if (kid === null || kid === undefined || kid === false) continue;
      n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return n;
  },
  esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); },
  num(v, d = 4) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-4 || a >= 1e6)) return v.toExponential(2);
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(a >= 100 ? 1 : d);
  },
  bytes(b) {
    if (b === null || b === undefined || b < 0) return '—';
    if (b > 1048576) return (b / 1048576).toFixed(1) + ' MB';
    if (b > 1024) return (b / 1024).toFixed(1) + ' KB';
    return b + ' B';
  },
  clock(ts) { return new Date(ts).toTimeString().slice(0, 8); },
  rel(iso) {
    if (!iso) return '—';
    const s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 60) return Math.max(0, Math.round(s)) + 's 前';
    if (s < 3600) return Math.round(s / 60) + 'm 前';
    if (s < 86400) return Math.round(s / 3600) + 'h 前';
    return Math.round(s / 86400) + 'd 前';
  },
  dur(a, b) {
    if (!a) return '—';
    const s = ((b ? new Date(b) : new Date()) - new Date(a)) / 1000;
    if (s < 0) return '—';
    if (s < 60) return s.toFixed(1) + 's';
    if (s < 3600) return Math.floor(s / 60) + 'm' + Math.round(s % 60) + 's';
    return Math.floor(s / 3600) + 'h' + Math.round((s % 3600) / 60) + 'm';
  },
  badge(status) { return UI.el('span', { class: 'badge ' + (status || 'queued') }, status || 'queued'); },
  toast(msg, kind = 'ok', ms = 3200) {
    const t = UI.el('div', { class: 'toast ' + kind }, msg);
    document.getElementById('toasts').append(t);
    setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .3s'; setTimeout(() => t.remove(), 320); }, ms);
  },
  err(e) { UI.toast(e && e.message ? e.message : String(e), 'err', 5000); },
  // modal({title, body: Node|Node[], actions:[{label,kind,onClick(close)}], wide}) -> close()
  modal({ title, body, actions = [], wide = false }) {
    const root = document.getElementById('modal-root');
    root.innerHTML = '';
    root.classList.add('open');
    const close = () => { root.classList.remove('open'); root.innerHTML = ''; document.removeEventListener('keydown', onKey); };
    const onKey = e => { if (e.key === 'Escape') close(); };
    document.addEventListener('keydown', onKey);
    const box = UI.el('div', { class: 'modal' + (wide ? ' wide' : '') },
      UI.el('div', { class: 'modal-h' }, title,
        UI.el('button', { class: 'btn sm ghost x', onclick: close }, '✕')),
      UI.el('div', { class: 'modal-b' }, body),
      actions.length ? UI.el('div', { class: 'modal-f' },
        actions.map(a => UI.el('button', {
          class: 'btn ' + (a.kind || ''), onclick: () => a.onClick ? a.onClick(close) : close(),
        }, a.label))) : null);
    root.append(UI.el('div', { class: 'modal-mask', onclick: close }), box);
    return close;
  },
  confirm(title, msg, label = '确认', kind = 'danger') {
    return new Promise(res => {
      UI.modal({
        title, body: UI.el('div', { style: { color: 'var(--muted)' } }, msg),
        actions: [
          { label: '取消', kind: 'ghost', onClick: c => { c(); res(false); } },
          { label, kind, onClick: c => { c(); res(true); } },
        ],
      });
    });
  },
  prompt(title, fields, submitLabel = '保存') {
    // fields: [{key,label,value,placeholder,type,help}] -> Promise<values|null>
    return new Promise(res => {
      const inputs = {};
      const body = fields.map(f => {
        const inp = UI.el('input', { class: 'input', type: f.type || 'text', value: f.value ?? '', placeholder: f.placeholder || '' });
        inputs[f.key] = inp;
        return UI.el('label', { class: 'fld' },
          UI.el('span', {}, f.label, f.help ? UI.el('span', { class: 'help', title: f.help }, '?') : null), inp);
      });
      UI.modal({
        title, body,
        actions: [
          { label: '取消', kind: 'ghost', onClick: c => { c(); res(null); } },
          {
            label: submitLabel, kind: 'primary', onClick: c => {
              const out = {};
              for (const [k, i] of Object.entries(inputs)) out[k] = i.value.trim();
              c(); res(out);
            },
          },
        ],
      });
      setTimeout(() => Object.values(inputs)[0]?.focus(), 30);
    });
  },
  // cols: [{label, num?, get(row)->Node|string, width?}]
  table(cols, rows, emptyMsg = '暂无数据') {
    if (!rows.length) return UI.el('div', { class: 'empty' }, UI.el('b', {}, emptyMsg), '没有可显示的记录');
    const t = UI.el('table', { class: 'tbl' },
      UI.el('thead', {}, UI.el('tr', {}, cols.map(c => UI.el('th', { class: c.num ? 'num' : '' }, c.label)))),
      UI.el('tbody', {}, rows.map(r => UI.el('tr', {}, cols.map(c => {
        const v = c.get(r);
        return UI.el('td', { class: (c.num ? 'num ' : '') + (c.cls || '') }, v);
      })))));
    return t;
  },
  download(name, text, mime = 'text/plain') {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: mime }));
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  },
  debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; },
  throttle(fn, ms) { let last = 0, t; return (...a) => { const now = Date.now(); clearTimeout(t); if (now - last >= ms) { last = now; fn(...a); } else t = setTimeout(() => { last = Date.now(); fn(...a); }, ms - (now - last)); }; },
  md(src) {
    // minimal markdown: headings, tables, code fences, lists, bold/inline code
    const esc = UI.esc;
    const lines = String(src || '').split('\n');
    let out = [], inCode = false, inTable = false, inList = false;
    const closeBlocks = () => {
      if (inTable) { out.push('</table>'); inTable = false; }
      if (inList) { out.push('</ul>'); inList = false; }
    };
    const inline = s => esc(s)
      .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
      .replace(/`([^`]+)`/g, '<code>$1</code>');
    for (const ln of lines) {
      if (ln.startsWith('```')) { closeBlocks(); inCode = !inCode; out.push(inCode ? '<pre>' : '</pre>'); continue; }
      if (inCode) { out.push(esc(ln)); continue; }
      if (/^\|/.test(ln)) {
        const cells = ln.split('|').slice(1, -1).map(c => c.trim());
        if (cells.every(c => /^:?-{2,}:?$/.test(c))) continue;
        if (!inTable) { closeBlocks(); out.push('<table>'); inTable = true; out.push('<tr>' + cells.map(c => '<th>' + inline(c) + '</th>').join('') + '</tr>'); }
        else out.push('<tr>' + cells.map(c => '<td>' + inline(c) + '</td>').join('') + '</tr>');
        continue;
      }
      if (inTable) { out.push('</table>'); inTable = false; }
      const h = ln.match(/^(#{1,4})\s+(.*)/);
      if (h) { closeBlocks(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); continue; }
      if (/^[-*]\s+/.test(ln)) { if (!inList) { out.push('<ul>'); inList = true; } out.push('<li>' + inline(ln.replace(/^[-*]\s+/, '')) + '</li>'); continue; }
      if (inList) { out.push('</ul>'); inList = false; }
      if (ln.trim() === '') continue;
      out.push('<p>' + inline(ln) + '</p>');
    }
    closeBlocks();
    if (inCode) out.push('</pre>');
    return out.join('\n');
  },
};