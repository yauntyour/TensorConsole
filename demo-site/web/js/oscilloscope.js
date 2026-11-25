// Gradient oscilloscope: shared step/rank store + weight heatmap sub-view.
// Arch (architecture dataflow) and Moe (routing analysis) read the same store.
const Osc = {
  order: [], rows: [], steps: [], ranks: [],
  view: [],            // indices into order after module/search filter
  matrix: [],          // [viewIdx][stepIdx] = l2 | null
  full: [],            // [stepIdx][orderIdx] = l2 | null (rank-aggregated, unfiltered)
  draw: [], drawH: 0,  // heatmap draw list: group headers + rows
  rank: -1, module: 'all', search: '', scale: 'log', labelMode: 'compact',
  sel: 0, step: 0, lo: -6, hi: 0,
  subview: 'arch', _subs: [], _hover: -1,
  playing: false, _timer: null, _base: null, _tip: null,
  ROW: 16, HEAD: 21, AXIS: 22, MINCELL: 8,

  sub(fn) { this._subs.push(fn); },
  emit(kind) { for (const fn of this._subs) fn(kind); },
  vecAt(i) { return this.full[i] || []; },
  l2(name, i = this.step) {
    const oi = this._idx ? this._idx.get(name) : undefined;
    return oi === undefined ? null : (this.full[i] || [])[oi] ?? null;
  },
  gut() { return this.labelMode === 'full' ? 306 : 176; },

  // compact semantic alias: decoder_layers.9.moe.experts.0.gate_up.weight -> L9·moe·e0·gate_up
  alias(name) {
    if (name === 'head_norm.weight') return 'head·norm';
    if (name === 'emb.weight') return 'emb·weight';
    let m = name.match(/^merge\.res\.(\d+)$/);
    if (m) return `merge·R${m[1]}`;
    m = name.match(/^decoder_layers\.(\d+)\.(.+)$/);
    if (!m) return name;
    const L = 'L' + m[1], r = m[2];
    const e = r.match(/^moe\.experts\.(\d+)\.(down|gate_up)\.weight$/);
    if (e) return `${L}·moe·e${e[1]}·${e[2]}`;
    const map = {
      'moe.gate.weight': 'moe·gate.w', 'moe.gate.bias': 'moe·gate.b', 'norm_moe.weight': 'moe·norm',
      'attn.out.weight': 'attn·out', 'attn.qkv.weight': 'attn·qkv', 'norm_attn.weight': 'attn·norm',
      'rt.w1.weight': 'rt·w1.w', 'rt.w1.bias': 'rt·w1.b', 'rt.w2.weight': 'rt·w2.w', 'rt.w2.bias': 'rt·w2.b',
      'rt.group_gate.weight': 'rt·gate.w', 'rt.group_gate.bias': 'rt·gate.b', 'norm_rt.weight': 'rt·norm',
    };
    return map[r] ? `${L}·${map[r]}` : `${L}·${r}`;
  },
  groupKey(o) {
    if (o.module === 'head' || o.module === 'emb') return o.module;
    if (o.module.startsWith('merge')) return 'merge';
    const m = o.module.match(/^L(\d+)\./);
    return m ? 'L' + m[1] : 'other';
  },
  groupLabel(key) {
    if (key === 'head') return 'head · tied';
    if (key === 'emb') return 'emb';
    if (key === 'merge') return 'birkhoff merge';
    if (key === 'other') return '其他参数';
    return `decoder L${key.slice(1)}`;
  },

  moduleColor(m) {
    if (m === 'head') return '#fb7185';
    if (m === 'emb') return '#60a5fa';
    if (m.startsWith('merge')) return '#fbbf24';
    if (m.endsWith('.moe')) return '#a78bfa';
    if (m.endsWith('.attn')) return '#2dd4bf';
    if (m.endsWith('.rt')) return '#34d399';
    return TC.dim;
  },
  // sequential flux ramp: cold navy -> electric blue -> teal -> acid -> amber -> white-hot
  RAMP: [[18, 26, 54], [37, 99, 235], [45, 212, 191], [163, 230, 53], [251, 191, 36], [255, 244, 214]],
  ramp(t) {
    const s = this.RAMP;
    t = Math.min(1, Math.max(0, t)) * (s.length - 1);
    const i = Math.min(s.length - 2, Math.floor(t)), f = t - i;
    const c = s[i].map((v, k) => Math.round(v + (s[i + 1][k] - v) * f));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  },
  rampCSS() { return `linear-gradient(90deg, ${this.RAMP.map(c => `rgb(${c.join(',')})`).join(',')})`; },
  norm(v) {
    if (v === null || v === undefined) return null;
    if (this.scale === 'lin') return this.hi > this.lo ? (v - this.lo) / (this.hi - this.lo) : 0;
    const l = Math.log10(Math.max(v, 1e-12));
    return (l - this.lo) / (this.hi - this.lo || 1);
  },

  setData(order, rows) {
    this.order = order || [];
    this._idx = new Map(this.order.map((o, i) => [o.name, i]));
    this.rows = rows || [];
    this.ranks = [...new Set(this.rows.map(r => r.rank ?? 0))].sort((a, b) => a - b);
    this.steps = [...new Set(this.rows.map(r => r.step))].sort((a, b) => a - b);
    const rs = document.getElementById('grad-rank');
    const cur = this.rank;
    rs.innerHTML = '';
    rs.append(UI.el('option', { value: '-1' }, '全部（均值）'));
    for (const r of this.ranks) rs.append(UI.el('option', { value: String(r) }, 'rank ' + r));
    rs.value = String(cur);
    const ms = document.getElementById('grad-module');
    const mods = [...new Set(this.order.map(o => o.module))];
    ms.innerHTML = '';
    ms.append(UI.el('option', { value: 'all' }, '全部模块'));
    for (const m of mods) ms.append(UI.el('option', { value: m }, m));
    ms.value = mods.includes(this.module) ? this.module : 'all';
    this.rebuild();
    this.emit('data');
  },

  rebuild() {
    const q = this.search.toLowerCase();
    this.view = this.order.map((o, i) => i).filter(i => {
      const o = this.order[i];
      if (this.module !== 'all' && o.module !== this.module) return false;
      if (q && !o.name.toLowerCase().includes(q) && !this.alias(o.name).toLowerCase().includes(q)) return false;
      return true;
    });
    const byStep = new Map();
    for (const r of this.rows) {
      if (this.rank >= 0 && (r.rank ?? 0) !== this.rank) continue;
      let e = byStep.get(r.step);
      if (!e) { e = { sum: new Array(this.order.length).fill(0), n: 0 }; byStep.set(r.step, e); }
      const norms = r.norms || [];
      for (let i = 0; i < norms.length; i++) {
        const n = norms[i];
        if (Array.isArray(n)) {
          const oi = this._idx.get(n[0]);
          if (oi !== undefined && typeof n[1] === 'number') e.sum[oi] += n[1];
        } else if (typeof n === 'number') e.sum[i] += n;
      }
      e.n++;
    }
    this.full = this.steps.map(s => {
      const e = byStep.get(s);
      if (!e || !e.n) return new Array(this.order.length).fill(null);
      return e.sum.map(v => v / e.n);
    });
    this.matrix = this.view.map(oi => this.full.map(row => row[oi]));
    let lo = Infinity, hi = -Infinity;
    for (const row of this.matrix) for (const v of row) {
      if (v === null || v <= 0) continue;
      const l = Math.log10(v);
      if (l < lo) lo = l;
      if (l > hi) hi = l;
    }
    if (!isFinite(lo)) { lo = -6; hi = 0; }
    if (hi - lo < 0.5) { hi = lo + 0.5; }
    this.lo = Math.floor(lo); this.hi = Math.ceil(hi);
    if (this.scale === 'lin') {
      let mlo = Infinity, mhi = -Infinity;
      for (const row of this.matrix) for (const v of row) if (v !== null) { mlo = Math.min(mlo, v); mhi = Math.max(mhi, v); }
      this.lo = isFinite(mlo) ? mlo : 0; this.hi = isFinite(mhi) ? mhi : 1;
    }
    // draw list with per-layer group headers
    this.draw = [];
    let y = 3, lastKey = null;
    for (let r = 0; r < this.view.length; r++) {
      const key = this.groupKey(this.order[this.view[r]]);
      if (key !== lastKey) {
        this.draw.push({ kind: 'head', key, y, h: this.HEAD });
        y += this.HEAD; lastKey = key;
      }
      this.draw.push({ kind: 'row', vi: r, y, h: this.ROW });
      y += this.ROW;
    }
    this.drawH = y + 3;
    this.sel = Math.min(this.sel, Math.max(0, this.view.length - 1));
    this.step = Math.min(this.step, Math.max(0, this.steps.length - 1));
    const sl = document.getElementById('grad-step');
    sl.max = String(Math.max(0, this.steps.length - 1));
    sl.value = String(this.step);
    document.getElementById('grad-step-label').textContent =
      this.steps.length ? `step ${this.steps[this.step]} (${this.step + 1}/${this.steps.length})` : '无数据';
    this.renderAll();
    this.emit('rebuild');
  },

  dpr() { return Math.min(2, window.devicePixelRatio || 1); },
  fit(canvas, w, h) {
    const d = this.dpr();
    canvas.width = Math.round(w * d); canvas.height = Math.round(h * d);
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(d, 0, 0, d, 0, 0);
    return ctx;
  },
  cellW() {
    const wrap = document.getElementById('grad-scroll');
    const avail = Math.max(200, wrap.clientWidth - this.gut() - 8);
    return Math.max(this.MINCELL, Math.floor(avail / Math.max(1, this.steps.length)));
  },

  setSubview(v) {
    this.subview = v;
    document.querySelectorAll('#grad-views button').forEach(b => b.classList.toggle('active', b.dataset.view === v));
    for (const id of ['arch', 'moe', 'heat'])
      document.getElementById('grad-sub-' + id).hidden = id !== v;
    if (App.page === 'job' && App.tab === 'grad') App.setHash(`job/${App.currentJob}/grad/${v}`);
    this.renderAll();
    this.emit('view');
  },

  renderAll() {
    this.renderLegend();
    if (this.subview === 'heat') this.renderHeatView();
    else if (this.subview === 'arch') Arch.render();
    else Moe.render();
  },

  renderLegend() {
    const box = document.getElementById('grad-legend');
    box.innerHTML = '';
    if (this.subview === 'arch') { Arch.renderLegend(box); return; }
    if (this.subview === 'moe') { Moe.renderLegend(box); return; }
    box.append(
      UI.el('span', {}, '低'), UI.el('span', { class: 'bar', style: { background: this.rampCSS() } }), UI.el('span', {}, '高'),
      UI.el('span', { class: 'mono', id: 'grad-range' },
        this.scale === 'log' ? `1e${this.lo} … 1e${this.hi}` : `${UI.num(this.lo, 2)} … ${UI.num(this.hi, 2)}`),
      UI.el('span', { class: 'lg-zero' }, UI.el('i', {}), '零梯度 / 未路由'));
  },

  renderHeatView() {
    const sc = document.getElementById('grad-scroll');
    let em = document.getElementById('grad-empty');
    if (!em) {
      em = UI.el('div', { class: 'empty', id: 'grad-empty' }, UI.el('b', {}, '无梯度数据'), '提交训练时开启「梯度示波器」或等待首个 step');
      sc.append(em);
    }
    const has = this.view.length && this.steps.length;
    em.hidden = !!has;
    for (const id of ['grad-heatmap', 'grad-axis', 'grad-wave'])
      document.getElementById(id).style.display = has ? 'block' : 'none';
    if (!has) {
      document.getElementById('grad-detail').innerHTML = '';
      document.getElementById('grad-stepinfo').innerHTML = '';
      return;
    }
    this.renderHeatmap();
    this.renderAxis();
    this.renderWave();
    this.renderDetail();
    this.renderStepInfo();
  },

  groupEnergy(key, si = this.step) {
    const vec = this.vecAt(si);
    let e = 0;
    for (let i = 0; i < this.order.length; i++)
      if (this.groupKey(this.order[i]) === key && vec[i] > 0) e += vec[i] * vec[i];
    return e;
  },

  renderHeatmap() {
    const GUT = this.gut();
    const cw = this.cellW();
    const w = GUT + cw * this.steps.length + 8;
    const h = this.drawH;
    const canvas = document.getElementById('grad-heatmap');
    const ctx = this.fit(canvas, w, h);
    ctx.fillStyle = TC.canvasBg; ctx.fillRect(0, 0, w, h);
    ctx.font = '10px ui-monospace, Consolas, monospace';
    ctx.textBaseline = 'middle';
    const vec = this.vecAt(this.step);
    let total = 0;
    for (const v of vec) if (v > 0) total += v * v;
    for (const d of this.draw) {
      if (d.kind === 'head') {
        ctx.fillStyle = TC.groupHead; ctx.fillRect(0, d.y, w, d.h - 2);
        ctx.fillStyle = TC.line; ctx.fillRect(0, d.y + d.h - 2, w, 1);
        const cy = d.y + (d.h - 2) / 2;
        ctx.font = '600 10px ui-monospace, Consolas, monospace';
        ctx.fillStyle = TC.muted;
        let hl = this.groupLabel(d.key).toUpperCase();
        const hmax = Math.floor((GUT - 84) / 6.1);
        if (hl.length > hmax) hl = hl.slice(0, hmax - 1) + '…';
        ctx.fillText(hl, 10, cy);
        const ge = this.groupEnergy(d.key);
        const share = total > 0 ? ge / total : 0;
        ctx.font = '10px ui-monospace, Consolas, monospace';
        ctx.fillStyle = TC.dim; ctx.textAlign = 'right';
        ctx.fillText(`能量 ${(share * 100).toFixed(1)}%`, GUT - 8, cy);
        ctx.textAlign = 'left';
        ctx.fillStyle = TC.panel3; ctx.fillRect(GUT, d.y + 6, cw * this.steps.length - 1, d.h - 14);
        ctx.fillStyle = this.moduleColor(d.key === 'other' ? 'other' : d.key === 'merge' ? 'merge.out' : d.key === 'head' ? 'head' : d.key === 'emb' ? 'emb' : d.key + '.rt');
        ctx.fillRect(GUT, d.y + 6, Math.max(1, (cw * this.steps.length - 1) * share), d.h - 14);
        continue;
      }
      const r = d.vi, o = this.order[this.view[r]];
      const y = d.y;
      if (r === this._hover) { ctx.fillStyle = 'rgba(45,212,191,.06)'; ctx.fillRect(0, y - 1, w, this.ROW - 1); }
      ctx.fillStyle = this.moduleColor(o.module);
      ctx.fillRect(0, y, 3, this.ROW - 3);
      ctx.fillStyle = r === this.sel ? '#2dd4bf' : TC.muted;
      let name = this.labelMode === 'full' ? o.name : this.alias(o.name);
      const maxChars = Math.floor((GUT - 16) / 6.1);
      if (name.length > maxChars) name = name.slice(0, maxChars - 1) + '…';
      ctx.fillText(name, 10, y + (this.ROW - 3) / 2);
      for (let c = 0; c < this.steps.length; c++) {
        const v = this.matrix[r][c];
        const x = GUT + c * cw;
        if (v === null) { ctx.fillStyle = TC.nullCell; }
        else if (v === 0) { ctx.fillStyle = '#3a1220'; }
        else ctx.fillStyle = this.ramp(this.norm(v));
        ctx.fillRect(x, y, cw - 1, this.ROW - 3);
        if (v === 0) { ctx.fillStyle = '#f87171'; ctx.fillRect(x, y + this.ROW - 5, cw - 1, 2); }
      }
      if (r === this.sel) {
        ctx.strokeStyle = 'rgba(45,212,191,.85)'; ctx.lineWidth = 1;
        ctx.strokeRect(0.5, y - 0.5, w - 1, this.ROW - 2);
      }
    }
    const x = GUT + this.step * cw;
    ctx.strokeStyle = TC.text; ctx.lineWidth = 1.4;
    ctx.strokeRect(x - 0.7, 1, cw + 0.4, h - 3);
    this._base = { w, h, cw, GUT };
  },

  renderAxis() {
    const { w, cw, GUT } = this._base || { w: 400, cw: 8, GUT: 158 };
    const canvas = document.getElementById('grad-axis');
    const ctx = this.fit(canvas, w, this.AXIS);
    ctx.fillStyle = TC.canvasBg; ctx.fillRect(0, 0, w, this.AXIS);
    ctx.font = '9.5px ui-monospace, Consolas, monospace';
    ctx.textBaseline = 'middle';
    const every = Math.max(1, Math.ceil(46 / cw));
    for (let c = 0; c < this.steps.length; c += every) {
      const x = GUT + c * cw;
      ctx.fillStyle = c === this.step ? '#2dd4bf' : TC.dim;
      ctx.fillText(String(this.steps[c]), x + 1, this.AXIS / 2);
      ctx.fillStyle = TC.line; ctx.fillRect(x, this.AXIS - 4, 1, 4);
    }
    ctx.fillStyle = TC.dim;
    ctx.fillText('step →', 10, this.AXIS / 2);
    document.getElementById('grad-axis-wrap').scrollLeft = document.getElementById('grad-scroll').scrollLeft;
  },

  renderWave() {
    const wrap = document.getElementById('grad-wave-wrap');
    const w = Math.max(240, wrap.clientWidth);
    const h = 148;
    const canvas = document.getElementById('grad-wave');
    const ctx = this.fit(canvas, w, h);
    ctx.fillStyle = TC.canvasBg; ctx.fillRect(0, 0, w, h);
    const n = this.view.length;
    const bw = w / n;
    const base = h - 18;
    ctx.fillStyle = TC.line; ctx.fillRect(0, base, w, 1);
    for (let i = 0; i < n; i++) {
      const v = this.matrix[i][this.step];
      const o = this.order[this.view[i]];
      let t = v === null ? 0 : (v === 0 ? 0 : this.norm(v));
      const bh = Math.max(v === null ? 0 : 2, t * (base - 12));
      ctx.fillStyle = i === this.sel ? TC.text : this.moduleColor(o.module);
      ctx.globalAlpha = i === this.sel ? 1 : 0.82;
      ctx.fillRect(i * bw + 0.5, base - bh, Math.max(1, bw - 1.2), bh);
      ctx.globalAlpha = 1;
      if (v === 0) { ctx.fillStyle = '#f87171'; ctx.fillRect(i * bw + 0.5, base + 2, Math.max(1, bw - 1.2), 3); }
    }
    ctx.font = '9.5px ui-monospace, Consolas, monospace';
    ctx.fillStyle = TC.dim;
    ctx.fillText('输出 → 输入', 6, h - 6);
    const sv = this.matrix[this.sel]?.[this.step];
    ctx.fillStyle = TC.muted;
    const label = this.alias(this.order[this.view[this.sel]]?.name || '');
    ctx.fillText(`${label}  l2=${sv === null || sv === undefined ? '—' : UI.num(sv, 5)}`, Math.min(w - 260, 90), h - 6);
  },

  renderDetail() {
    const box = document.getElementById('grad-detail');
    box.innerHTML = '';
    if (!this.view.length) return;
    const o = this.order[this.view[this.sel]];
    const series = this.matrix[this.sel] || [];
    const vals = series.filter(v => v !== null);
    const zeros = vals.filter(v => v === 0).length;
    const last = [...vals].reverse().find(v => v !== undefined);
    const mn = vals.length ? Math.min(...vals) : null;
    const mx = vals.length ? Math.max(...vals) : null;
    const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    box.append(
      UI.el('div', { class: 'pname' }, o.name),
      UI.el('div', { class: 'kv-chips', style: { margin: '8px 0' } },
        UI.el('span', { class: 'tag', style: { color: this.moduleColor(o.module) } }, this.alias(o.name)),
        UI.el('span', { class: 'tag' }, 'depth ' + o.depth),
        zeros ? UI.el('span', { class: 'zero-badge' }, `零梯度 ×${zeros}`) : null),
      UI.el('div', { class: 'kv' },
        UI.el('span', { class: 'k' }, '当前 l2'), UI.el('span', { class: 'v' }, UI.num(series[this.step], 6)),
        UI.el('span', { class: 'k' }, '末值'), UI.el('span', { class: 'v' }, UI.num(last, 6)),
        UI.el('span', { class: 'k' }, 'min / max'), UI.el('span', { class: 'v' }, `${UI.num(mn, 3)} / ${UI.num(mx, 3)}`),
        UI.el('span', { class: 'k' }, '均值'), UI.el('span', { class: 'v' }, UI.num(mean, 6)),
        UI.el('span', { class: 'k' }, '采样步数'), UI.el('span', { class: 'v' }, String(vals.length))));
    const spark = UI.el('canvas', { style: { marginTop: '10px', width: '100%' } });
    box.append(spark);
    const w = box.clientWidth - 0 || 260, h = 64;
    const ctx = this.fit(spark, w, h);
    ctx.fillStyle = TC.canvasBg; ctx.fillRect(0, 0, w, h);
    if (vals.length) {
      const lmin = Math.log10(Math.max(mn, 1e-12)), lmax = Math.log10(Math.max(mx, 1e-12));
      const span = (lmax - lmin) || 1;
      ctx.strokeStyle = '#2dd4bf'; ctx.lineWidth = 1.4; ctx.beginPath();
      series.forEach((v, i) => {
        if (v === null) return;
        const x = 4 + (i / Math.max(1, series.length - 1)) * (w - 8);
        const y = h - 8 - ((Math.log10(Math.max(v, 1e-12)) - lmin) / span) * (h - 16);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
      const sx = 4 + (this.step / Math.max(1, series.length - 1)) * (w - 8);
      ctx.fillStyle = TC.text; ctx.fillRect(sx - 0.5, 2, 1, h - 4);
    }
    spark.addEventListener('click', ev => {
      const r = spark.getBoundingClientRect();
      const i = Math.round(((ev.clientX - r.left - 4) / (r.width - 8)) * (series.length - 1));
      this.setStep(Math.max(0, Math.min(series.length - 1, i)));
    });
  },

  renderStepInfo() {
    const box = document.getElementById('grad-stepinfo');
    box.innerHTML = '';
    if (!this.steps.length) return;
    const s = this.steps[this.step];
    const col = this.matrix.map(r => r[this.step]);
    const vals = col.filter(v => v !== null);
    const zeros = vals.filter(v => v === 0).length;
    const row = this.rows.find(r => r.step === s && (this.rank < 0 || (r.rank ?? 0) === this.rank));
    const top = this.view.map((oi, i) => ({ name: this.alias(this.order[oi].name), v: col[i] }))
      .filter(x => x.v !== null).sort((a, b) => b.v - a.v).slice(0, 5);
    box.append(
      UI.el('div', { class: 'kv' },
        UI.el('span', { class: 'k' }, 'step'), UI.el('span', { class: 'v' }, String(s)),
        UI.el('span', { class: 'k' }, '时间'), UI.el('span', { class: 'v' }, row?.t ? UI.clock(row.t) : '—'),
        UI.el('span', { class: 'k' }, '权重数'), UI.el('span', { class: 'v' }, String(vals.length)),
        UI.el('span', { class: 'k' }, '零梯度'), UI.el('span', { class: 'v', style: zeros ? { color: 'var(--red)' } : {} }, String(zeros))),
      UI.el('div', { style: { marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '4px' } },
        top.map(t => UI.el('div', { style: { display: 'flex', gap: '8px', fontSize: '11px' } },
          UI.el('span', { class: 'mono', style: { color: 'var(--muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: '1' } }, t.name),
          UI.el('span', { class: 'mono', style: { color: 'var(--accent)' } }, UI.num(t.v, 4))))));
  },

  setStep(i) {
    this.step = Math.max(0, Math.min(this.steps.length - 1, i));
    document.getElementById('grad-step').value = String(this.step);
    document.getElementById('grad-step-label').textContent = `step ${this.steps[this.step] ?? '—'} (${this.step + 1}/${this.steps.length})`;
    if (this.subview === 'heat') {
      this.renderHeatmap(); this.renderAxis(); this.renderWave(); this.renderDetail(); this.renderStepInfo();
    } else this.renderAll();
    this.emit('step');
  },
  select(i) { this.sel = i; this.renderHeatmap(); this.renderWave(); this.renderDetail(); },
  play(on) {
    this.playing = on === undefined ? !this.playing : on;
    document.getElementById('grad-play').textContent = this.playing ? '⏸ 暂停' : '▶ 回放';
    clearInterval(this._timer);
    if (this.playing) this._timer = setInterval(() => {
      this.setStep((this.step + 1) % Math.max(1, this.steps.length));
    }, 260);
  },

  bind() {
    const $ = id => document.getElementById(id);
    document.querySelectorAll('#grad-views button').forEach(b =>
      b.addEventListener('click', () => this.setSubview(b.dataset.view)));
    $('grad-rank').addEventListener('change', e => { this.rank = +e.target.value; this.rebuild(); });
    $('grad-module').addEventListener('change', e => { this.module = e.target.value; this.rebuild(); });
    $('grad-search').addEventListener('input', UI.debounce(e => { this.search = e.target.value; this.rebuild(); }, 200));
    $('grad-scale').addEventListener('change', e => { this.scale = e.target.value; this.rebuild(); });
    $('grad-labels').addEventListener('change', e => { this.labelMode = e.target.value; this.rebuild(); });
    $('grad-step').addEventListener('input', e => this.setStep(+e.target.value));
    $('grad-prev').addEventListener('click', () => this.setStep(this.step - 1));
    $('grad-next').addEventListener('click', () => this.setStep(this.step + 1));
    $('grad-play').addEventListener('click', () => this.play());
    $('grad-reload').addEventListener('click', () => App.reloadGrad());
    $('grad-scroll').addEventListener('scroll', () => {
      $('grad-axis-wrap').scrollLeft = $('grad-scroll').scrollLeft;
    });
    const tip = this._tip = UI.el('div', { class: 'ctip' });
    document.body.append(tip);
    const locate = ev => {
      const canvas = $('grad-heatmap');
      const r = canvas.getBoundingClientRect();
      const x = ev.clientX - r.left, y = ev.clientY - r.top;
      if (!this._base) return null;
      const col = Math.floor((x - this._base.GUT) / this._base.cw);
      if (col < 0 || col >= this.steps.length) return null;
      const d = this.draw.find(q => y >= q.y && y < q.y + q.h);
      if (!d || d.kind !== 'row') return null;
      return { row: d.vi, col };
    };
    $('grad-heatmap').addEventListener('mousemove', ev => {
      const p = locate(ev);
      if (!p) { tip.style.display = 'none'; if (this._hover !== -1) { this._hover = -1; this.renderHeatmap(); } return; }
      if (this._hover !== p.row) { this._hover = p.row; this.renderHeatmap(); }
      const o = this.order[this.view[p.row]];
      const v = this.matrix[p.row][p.col];
      tip.innerHTML = `<b>${UI.esc(o.name)}</b><span class="t2">${UI.esc(this.alias(o.name))} · step ${this.steps[p.col]} · l2 ${v === null ? '—' : UI.num(v, 6)}${v === 0 ? ' <em>ZERO · 未路由/冻结</em>' : ''}</span>`;
      tip.style.display = 'block';
      tip.style.left = Math.min(window.innerWidth - 300, ev.clientX + 14) + 'px';
      tip.style.top = (ev.clientY + 14) + 'px';
    });
    $('grad-heatmap').addEventListener('mouseleave', () => {
      tip.style.display = 'none';
      if (this._hover !== -1) { this._hover = -1; this.renderHeatmap(); }
    });
    $('grad-heatmap').addEventListener('click', ev => {
      const p = locate(ev);
      if (!p) return;
      this.sel = p.row; this.setStep(p.col);
    });
    $('grad-wave').addEventListener('click', ev => {
      const r = ev.target.getBoundingClientRect();
      const i = Math.floor(((ev.clientX - r.left) / r.width) * this.view.length);
      if (i >= 0 && i < this.view.length) this.select(i);
    });
    window.addEventListener('resize', UI.debounce(() => { if (this.view.length) this.renderAll(); }, 150));
  },
};