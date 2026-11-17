// Timing analysis: chrome-trace flame graph + eng_prof operator bars + sortable table
const Flame = {
  profile: null, zoom: null, sortKey: 'total_ms', sortDir: -1, _tip: null, _spans: [],

  hashColor(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
    const hue = Math.abs(h) % 360;
    return `hsl(${hue} 46% ${42 + (Math.abs(h >> 8) % 14)}%)`;
  },
  phaseColor(name) {
    if (/bwd|backward/.test(name)) return '#a78bfa';
    if (/fwd|forward/.test(name)) return '#2dd4bf';
    if (/opt/.test(name)) return '#fbbf24';
    if (/ce|head/.test(name)) return '#f472b6';
    return '#60a5fa';
  },
  fit(canvas, w, h) {
    const d = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * d); canvas.height = Math.round(h * d);
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(d, 0, 0, d, 0, 0);
    return ctx;
  },

  setData(p) {
    this.profile = p || null;
    this.zoom = null;
    this._spans = (p && p.spans) || [];
    const src = document.getElementById('prof-source');
    src.textContent = p ? (p.source === 'eng_prof' ? '引擎 ENG_PROF' : 'Chrome Trace') : '无数据';
    document.getElementById('prof-total').textContent = p ? `total ${UI.num(p.total_ms, 4)} ms` : '';
    document.getElementById('flame-reset').disabled = !this._spans.length;
    this.draw();
    this.table();
  },

  draw() {
    const wrap = document.getElementById('flame-wrap');
    const canvas = document.getElementById('flame-canvas');
    const w = Math.max(280, wrap.clientWidth);
    if (!this.profile) { canvas.height = 0; canvas.style.height = '0px'; return; }
    if (this._spans.length) this.drawFlame(canvas, w);
    else this.drawBars(canvas, w);
  },

  drawFlame(canvas, w) {
    const spans = this._spans;
    const t0 = this.zoom ? this.zoom[0] : Math.min(...spans.map(s => s.ts_us));
    const t1 = this.zoom ? this.zoom[1] : Math.max(...spans.map(s => s.ts_us + s.dur_us));
    const span = (t1 - t0) || 1;
    const threads = [...new Set(spans.map(s => s.pid + ':' + s.tid))];
    const rowH = 20, gap = 6;
    const depthOf = s => threads.indexOf(s.pid + ':' + s.tid);
    const maxDepth = Math.max(...spans.map(s => s.depth)) + 1;
    const h = threads.length * (maxDepth * rowH + gap) + 26;
    const ctx = this.fit(canvas, w, h);
    ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, w, h);
    ctx.font = '10px ui-monospace, Consolas, monospace';
    ctx.textBaseline = 'middle';
    this._rects = [];
    threads.forEach((th, ti) => {
      const yBase = ti * (maxDepth * rowH + gap) + 4;
      ctx.fillStyle = '#64748b';
      ctx.fillText('thread ' + th, 6, yBase + 6);
      for (const s of spans) {
        if (depthOf(s) !== ti) continue;
        const x = ((s.ts_us - t0) / span) * w;
        const sw = (s.dur_us / span) * w;
        if (x + sw < 0 || x > w || sw < 0.4) continue;
        const y = yBase + 14 + s.depth * rowH;
        ctx.fillStyle = this.hashColor(s.name);
        ctx.fillRect(Math.max(0, x), y, Math.min(w, x + sw) - Math.max(0, x), rowH - 2);
        if (sw > 46) {
          ctx.save();
          ctx.beginPath(); ctx.rect(Math.max(0, x) + 3, y, sw - 6, rowH - 2); ctx.clip();
          ctx.fillStyle = '#0a0d12';
          ctx.fillText(`${s.name}  ${UI.num(s.dur_us / 1000, 2)}ms`, Math.max(0, x) + 5, y + (rowH - 2) / 2);
          ctx.restore();
        }
        this._rects.push({ x: Math.max(0, x), y, w: Math.min(w, x + sw) - Math.max(0, x), h: rowH - 2, s });
      }
    });
    // time ruler
    ctx.fillStyle = '#64748b';
    for (let i = 0; i <= 8; i++) {
      const t = t0 + (span * i) / 8;
      ctx.fillText(UI.num((t - t0) / 1000, 1) + 'ms', (w * i) / 8 + 2, h - 8);
    }
  },

  drawBars(canvas, w) {
    const sum = (this.profile.summary || []).slice(0, 24);
    const rowH = 22;
    const h = sum.length * rowH + 10;
    const ctx = this.fit(canvas, w, h);
    ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, w, h);
    ctx.font = '10.5px ui-monospace, Consolas, monospace';
    ctx.textBaseline = 'middle';
    const max = Math.max(...sum.map(s => s.total_ms), 1e-9);
    const labelW = Math.min(240, w * 0.34);
    this._rects = [];
    sum.forEach((s, i) => {
      const y = i * rowH + 5;
      ctx.fillStyle = '#93a1b0';
      let nm = s.name;
      if (nm.length > labelW / 6.2) nm = nm.slice(0, Math.floor(labelW / 6.2) - 1) + '…';
      ctx.fillText(nm, 8, y + 8);
      const bw = (s.total_ms / max) * (w - labelW - 130);
      ctx.fillStyle = s.kind === 'phase' ? this.phaseColor(s.name) : '#3b82f6';
      ctx.fillRect(labelW, y + 3, Math.max(2, bw), 12);
      ctx.fillStyle = '#64748b';
      ctx.fillText(`${UI.num(s.total_ms, 3)}ms · ${UI.num(s.pct, 1)}% · n=${s.count}`, labelW + Math.max(2, bw) + 8, y + 9);
      this._rects.push({ x: labelW, y: y + 3, w: Math.max(2, bw), h: 12, s: { name: s.name, dur_us: s.total_ms * 1000, summary: s } });
    });
  },

  table() {
    const t = document.getElementById('prof-table');
    t.innerHTML = '';
    const sum = [...((this.profile && this.profile.summary) || [])];
    const k = this.sortKey, dir = this.sortDir;
    sum.sort((a, b) => ((a[k] ?? 0) - (b[k] ?? 0)) * dir);
    const max = Math.max(...sum.map(s => s.total_ms || 0), 1e-9);
    const cols = [
      ['name', '名称'], ['kind', '类别'], ['count', '次数'],
      ['total_ms', '总耗时 ms'], ['avg_ms', '平均 ms'], ['pct', '占比'],
    ];
    const head = UI.el('tr', {}, cols.map(([key, label]) => UI.el('th', {
      class: key === 'name' || key === 'kind' ? '' : 'num',
      style: { cursor: 'pointer' },
      onclick: () => { if (this.sortKey === key) this.sortDir *= -1; else { this.sortKey = key; this.sortDir = -1; } this.table(); },
    }, label + (this.sortKey === key ? (this.sortDir < 0 ? ' ↓' : ' ↑') : ''))));
    const body = sum.map(s => UI.el('tr', {},
      UI.el('td', { class: 'mono' }, s.name),
      UI.el('td', {}, UI.el('span', { class: 'tag' }, s.kind || s.cat || 'op')),
      UI.el('td', { class: 'num' }, String(s.count ?? '—')),
      UI.el('td', { class: 'num' }, UI.num(s.total_ms, 3),
        UI.el('div', {}, UI.el('span', { class: 'prof-bar', style: { width: Math.round((s.total_ms / max) * 90) + 'px', background: s.kind === 'phase' ? this.phaseColor(s.name) : 'var(--blue)' } }))),
      UI.el('td', { class: 'num' }, UI.num(s.avg_ms, 4)),
      UI.el('td', { class: 'num' }, UI.num(s.pct, 1) + '%')));
    t.append(UI.el('thead', {}, head), UI.el('tbody', {}, body));
    if (!sum.length) t.append(UI.el('tbody', {}, UI.el('tr', {}, UI.el('td', { colspan: 6 }, UI.el('div', { class: 'empty' }, '暂无耗时数据 — 训练时开启「耗时分析」或上传 Chrome Trace')))));
  },

  bind() {
    const $ = id => document.getElementById(id);
    const tip = this._tip = UI.el('div', {
      style: {
        position: 'fixed', zIndex: 300, pointerEvents: 'none', display: 'none',
        background: 'var(--panel-2)', border: '1px solid var(--line)', borderRadius: '8px',
        padding: '6px 10px', fontSize: '11px', fontFamily: 'var(--mono)', boxShadow: 'var(--shadow)',
      },
    });
    document.body.append(tip);
    const hit = ev => {
      const r = $('flame-canvas').getBoundingClientRect();
      const x = ev.clientX - r.left, y = ev.clientY - r.top;
      return (this._rects || []).find(q => x >= q.x && x <= q.x + q.w && y >= q.y && y <= q.y + q.h);
    };
    $('flame-canvas').addEventListener('mousemove', ev => {
      const q = hit(ev);
      if (!q) { tip.style.display = 'none'; return; }
      const s = q.s;
      const extra = s.summary ? `n=${s.summary.count} · avg ${UI.num(s.summary.avg_ms, 4)}ms · ${UI.num(s.summary.pct, 1)}%` : `${UI.num(s.dur_us / 1000, 3)}ms · depth ${s.depth}`;
      tip.innerHTML = `<b style="color:var(--accent)">${UI.esc(s.name)}</b><br>${extra}`;
      tip.style.display = 'block';
      tip.style.left = Math.min(window.innerWidth - 260, ev.clientX + 14) + 'px';
      tip.style.top = (ev.clientY + 14) + 'px';
    });
    $('flame-canvas').addEventListener('mouseleave', () => { tip.style.display = 'none'; });
    $('flame-canvas').addEventListener('click', ev => {
      const q = hit(ev);
      if (q && q.s && q.s.ts_us !== undefined) this.zoom = [q.s.ts_us, q.s.ts_us + q.s.dur_us];
      this.draw();
    });
    $('flame-canvas').addEventListener('dblclick', () => { this.zoom = null; this.draw(); });
    $('flame-reset').addEventListener('click', () => { this.zoom = null; this.draw(); });
    $('prof-reload').addEventListener('click', () => App.reloadProfile());
    $('prof-upload-btn').addEventListener('click', () => $('prof-upload').click());
    $('prof-upload').addEventListener('change', async ev => {
      const f = ev.target.files[0];
      if (!f) return;
      try {
        const j = JSON.parse(await f.text());
        await API.post('/api/jobs/' + App.currentJob + '/profile', j);
        UI.toast('Trace 已上传并解析');
        App.reloadProfile();
      } catch (e) { UI.err(e); }
      ev.target.value = '';
    });
    window.addEventListener('resize', UI.debounce(() => this.draw(), 150));
  },
};