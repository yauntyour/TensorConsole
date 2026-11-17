// MoE routing analysis: layer x expert load/activation matrix, balance trends,
// coverage tables. Load is a gradient-energy proxy (zero grad = unrouted expert).
const Moe = {
  metric: 'load', window: 'step', sel: null, selExpert: null,
  cache: null, visible: false, _chart: null, _tip: null,

  invalidate() { this.cache = null; if (this.visible) this.render(); },

  compute() {
    if (this.cache) return this.cache;
    const S = Arch.meta.S, E = Arch.meta.E, T = Osc.full.length;
    const cells = [];
    for (let li = 0; li < S; li++) {
      const row = [];
      for (let e = 0; e < E; e++) {
        const dn = `decoder_layers.${li}.moe.experts.${e}.down.weight`;
        const up = `decoder_layers.${li}.moe.experts.${e}.gate_up.weight`;
        const i0 = Osc._idx.get(dn), i1 = Osc._idx.get(up);
        const gs = Osc.full.map(vec => {
          const a = i0 === undefined ? null : vec[i0], b = i1 === undefined ? null : vec[i1];
          if (a === null && b === null) return null;
          return Math.sqrt((a || 0) ** 2 + (b || 0) ** 2);
        });
        const act = gs.map(g => g !== null && g > 0);
        const pw = gs.map(g => Math.pow(g || 0, Arch.pow));
        const shares = new Array(T).fill(0);
        row.push({ li, e, gs, act, pw, shares, freq: 0, meanG: 0, meanShare: 0, maxShare: 0, streak: 0 });
      }
      // normalize shares per step within the layer
      for (let s = 0; s < T; s++) {
        let sum = 0;
        for (const c of row) sum += c.pw[s];
        for (const c of row) c.shares[s] = sum > 0 ? c.pw[s] / sum : 0;
      }
      for (const c of row) {
        const na = c.act.filter(Boolean).length;
        c.freq = T ? na / T : 0;
        c.meanG = c.gs.reduce((a, b) => a + (b || 0), 0) / Math.max(1, T);
        c.meanShare = c.shares.reduce((a, b) => a + b, 0) / Math.max(1, T);
        c.maxShare = Math.max(0, ...c.shares);
        let st = 0, best = 0;
        for (const a of c.act) { if (a) st = 0; else { st++; best = Math.max(best, st); } }
        c.streak = best;
        c.na = na;
      }
      cells.push(row);
    }
    const trend = { eff: [], nH: [], cv: [], act: [] };
    for (let s = 0; s < T; s++) {
      let eff = 0, nH = 0, cv = 0, na = 0, n = 0;
      for (const row of cells) {
        const sh = row.map(c => c.shares[s]);
        const sum = sh.reduce((a, b) => a + b, 0);
        let H = 0;
        for (const v of sh) if (v > 0) H -= v * Math.log(v);
        const E = row.length || 1;
        const mean = 1 / E;
        eff += Math.exp(H); nH += E > 1 ? H / Math.log(E) : 1;
        cv += Math.sqrt(sh.reduce((a, v) => a + (v - mean) ** 2, 0) / E) / mean;
        na += row.filter(c => c.act[s]).length; n += E;
      }
      const S = cells.length || 1;
      trend.eff.push(eff / S); trend.nH.push(nH / S); trend.cv.push(cv / S); trend.act.push(n ? na / n * 100 : 0);
    }
    this.cache = { cells, trend, S, E, T };
    return this.cache;
  },

  value(c) {
    const s = Osc.step;
    if (this.metric === 'active') return this.window === 'step' ? (c.act[s] ? 1 : 0) : c.freq;
    if (this.metric === 'grad') return this.window === 'step' ? (c.gs[s] || 0) : c.meanG;
    if (this.window === 'step') return c.shares[s];
    if (this.window === 'freq') return c.freq;
    return c.meanShare;
  },
  cellT(c, max) {
    if (this.metric === 'grad') return Osc.norm(this.value(c)) ?? 0;
    if (this.metric === 'active') return this.value(c) ? 0.35 + 0.6 * this.value(c) : 0.02;
    const v = this.value(c);
    return max > 0 ? v / max : 0;
  },

  render() {
    this.visible = !document.getElementById('grad-sub-moe').hidden;
    if (!this.visible) return;
    const has = Osc.steps.length && Arch.meta.S && Arch.meta.E;
    document.getElementById('moe-empty').hidden = !!has;
    document.getElementById('moe-body').hidden = !has;
    if (!has) { if (this._chart) { this._chart.destroy(); this._chart = null; } return; }
    const { cells, S, E, T } = this.compute();
    document.getElementById('moe-matrix-sub').textContent =
      `${S} 层 × ${E} 专家 · ${this.window === 'step' ? 'step ' + Osc.steps[Osc.step] : this.window === 'freq' ? '全窗口 ' + T + ' step 路由频率' : '全窗口 ' + T + ' step 均值'} · p=${Arch.pow.toFixed(2)}`;
    this.renderMatrix(cells, S, E);
    this.renderTrend();
    this.renderExperts(cells, S, E, T);
    this.renderLayers(cells, S, E);
    this.renderDetail(cells);
  },

  renderMatrix(cells, S, E) {
    const wrap = document.getElementById('moe-matrix-wrap');
    const w = Math.max(320, wrap.clientWidth);
    const lab = 56, right = 62, top = 22, bot = 20;
    const cw = Math.max(24, Math.min(120, (w - lab - right - 8) / E));
    const rh = cw >= 60 ? 28 : 26;
    const h = top + S * rh + bot;
    const canvas = document.getElementById('moe-matrix');
    const ctx = Osc.fit(canvas, lab + E * cw + right, h);
    ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, lab + E * cw + right, h);
    ctx.textBaseline = 'middle';
    ctx.font = '9.5px ui-monospace, Consolas, monospace';
    for (let e = 0; e < E; e++) {
      ctx.fillStyle = this.selExpert === e ? '#f472b6' : '#64748b';
      ctx.textAlign = 'center';
      ctx.fillText('e' + e, lab + e * cw + cw / 2, top / 2);
    }
    ctx.textAlign = 'left';
    this._mrects = [];
    for (let li = 0; li < S; li++) {
      const y = top + li * rh;
      ctx.fillStyle = Arch.layerHue(li);
      ctx.font = '600 10px ui-monospace, Consolas, monospace';
      ctx.fillText('L' + li, 8, y + rh / 2);
      const row = cells[li];
      const max = Math.max(1e-9, ...row.map(c => this.value(c)));
      let na = 0;
      for (let e = 0; e < E; e++) {
        const c = row[e];
        const x = lab + e * cw;
        const v = this.value(c);
        const dead = this.metric === 'active' ? v === 0 : (this.window === 'step' ? !c.act[Osc.step] : c.freq === 0);
        ctx.fillStyle = dead ? '#2a0f18' : Osc.ramp(this.cellT(c, max));
        ctx.fillRect(x + 1, y + 2, cw - 2, rh - 4);
        if (dead) {
          ctx.strokeStyle = 'rgba(248,113,113,.7)'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(x + 4, y + 5); ctx.lineTo(x + cw - 4, y + rh - 5); ctx.stroke();
        } else if (this.metric === 'load' && v * 100 < Arch.hungry) {
          ctx.strokeStyle = '#fbbf24'; ctx.lineWidth = 1;
          ctx.strokeRect(x + 1.5, y + 2.5, cw - 3, rh - 5);
        }
        if (this.sel && this.sel.li === li && this.sel.e === e) {
          ctx.strokeStyle = '#e8eef4'; ctx.lineWidth = 1.6;
          ctx.strokeRect(x + 0.8, y + 1.8, cw - 1.6, rh - 3.6);
        }
        if (cw >= 34 && !dead) {
          ctx.fillStyle = this.cellT(c, max) > 0.55 ? '#0a0e14' : '#cbd5e1';
          ctx.textAlign = 'center';
          ctx.font = '8.5px ui-monospace, Consolas, monospace';
          ctx.fillText(this.metric === 'grad' ? UI.num(v, 2) : (v * 100 < 0.5 ? '<1%' : (v * 100).toFixed(0) + '%'), x + cw / 2, y + rh / 2);
          ctx.font = '9.5px ui-monospace, Consolas, monospace';
          ctx.textAlign = 'left';
        }
        if (c.act[Osc.step]) na++;
        this._mrects.push({ x, y, w: cw, h: rh, c });
      }
      ctx.fillStyle = na === E ? '#34d399' : na === 0 ? '#f87171' : '#93a1b0';
      ctx.fillText(`${na}/${E}`, lab + E * cw + 10, y + rh / 2);
    }
    ctx.fillStyle = '#4b5a6b';
    ctx.font = '9px ui-monospace, Consolas, monospace';
    for (let e = 0; e < E; e++) {
      const f = cells.reduce((a, row) => a + row[e].freq, 0) / S;
      ctx.textAlign = 'center';
      ctx.fillStyle = f === 0 ? '#f87171' : f < 0.3 ? '#fbbf24' : '#64748b';
      ctx.fillText((f * 100).toFixed(0) + '%', lab + e * cw + cw / 2, top + S * rh + bot / 2);
    }
    ctx.textAlign = 'left';
    ctx.fillStyle = '#4b5a6b';
    ctx.fillText('路由频率', 6, top + S * rh + bot / 2);
  },

  renderTrend() {
    const box = document.getElementById('moe-trend');
    const { trend, T } = this.compute();
    const defs = [
      { label: '有效专家数 exp(H)', color: '#a78bfa', key: 'eff' },
      { label: '归一化熵', color: '#2dd4bf', key: 'nH' },
      { label: '负载 CV', color: '#fbbf24', key: 'cv' },
      { label: '激活率 %', color: '#f472b6', key: 'act', scale: 'pct' },
    ];
    if (!this._chart) this._chart = Charts.line(box, defs, { height: 300 });
    this._chart.setData(Osc.steps, defs.map(d => trend[d.key]));
  },

  renderExperts(cells, S, E, T) {
    const rows = [];
    for (let e = 0; e < E; e++) {
      const cs = cells.map(row => row[e]);
      const freq = cs.reduce((a, c) => a + c.freq, 0) / S;
      const meanLoad = cs.reduce((a, c) => a + c.meanShare, 0) / S;
      const peak = Math.max(...cs.map(c => c.maxShare));
      const streak = Math.max(...cs.map(c => c.streak));
      const state = freq === 0 ? ['死亡', 'var(--red)'] : freq < 0.25 ? ['饥饿', 'var(--amber)'] : freq < 0.7 ? ['间歇', 'var(--blue)'] : ['活跃', 'var(--green)'];
      rows.push({ e, freq, meanLoad, peak, streak, state });
    }
    const box = document.getElementById('moe-experts');
    box.innerHTML = '';
    box.append(UI.table([
      { label: '专家', get: r => UI.el('b', { class: 'mono', style: { color: this.selExpert === r.e ? '#f472b6' : 'var(--text)' } }, 'e' + r.e) },
      { label: '状态', get: r => UI.el('span', { class: 'tag', style: { color: r.state[1] } }, r.state[0]) },
      { label: '路由频率', num: true, get: r => (r.freq * 100).toFixed(1) + '%' },
      { label: '平均负载', num: true, get: r => (r.meanLoad * 100).toFixed(2) + '%' },
      { label: '峰值负载', num: true, get: r => (r.peak * 100).toFixed(1) + '%' },
      { label: '最长未路由', num: true, get: r => UI.el('span', { style: r.streak > 3 ? { color: 'var(--red)' } : {} }, r.streak + ' step') },
    ], rows, '无专家数据'));
    box.querySelectorAll('tbody tr').forEach((tr, i) => {
      tr.style.cursor = 'pointer';
      tr.addEventListener('click', () => { this.selExpert = this.selExpert === rows[i].e ? null : rows[i].e; this.render(); });
    });
  },

  renderLayers(cells, S, E) {
    const s = Osc.step;
    const rows = cells.map((row, li) => {
      const sh = row.map(c => c.shares[s]);
      let H = 0;
      for (const v of sh) if (v > 0) H -= v * Math.log(v);
      const na = row.filter(c => c.act[s]).length;
      const mean = 1 / E;
      const cv = Math.sqrt(sh.reduce((a, v) => a + (v - mean) ** 2, 0) / E) / mean;
      const top = sh.indexOf(Math.max(...sh));
      const wf = row.reduce((a, c) => a + c.freq, 0) / E;
      return { li, na, eff: Math.exp(H), nH: E > 1 ? H / Math.log(E) : 1, cv, top, wf };
    });
    const box = document.getElementById('moe-layers');
    box.innerHTML = '';
    box.append(UI.table([
      { label: '层', get: r => UI.el('b', { class: 'mono', style: { color: Arch.layerHue(r.li) } }, 'L' + r.li) },
      { label: '激活', num: true, get: r => UI.el('span', { style: r.na === 0 ? { color: 'var(--red)' } : r.na < E ? { color: 'var(--amber)' } : { color: 'var(--green)' } }, `${r.na}/${E}`) },
      { label: 'exp(H)', num: true, get: r => r.eff.toFixed(2) },
      { label: '归一化熵', num: true, get: r => r.nH.toFixed(3) },
      { label: 'CV', num: true, get: r => UI.el('span', { style: r.cv > 1 ? { color: 'var(--amber)' } : {} }, r.cv.toFixed(2)) },
      { label: '最热专家', get: r => UI.el('span', { class: 'mono' }, 'e' + r.top) },
      { label: '窗口路由率', num: true, get: r => (r.wf * 100).toFixed(0) + '%' },
    ], rows, '无层数据'));
    box.querySelectorAll('tbody tr').forEach((tr, i) => {
      tr.style.cursor = 'pointer';
      tr.addEventListener('click', () => {
        Arch.focus = String(rows[i].li);
        document.getElementById('arch-focus').value = Arch.focus;
        Arch.sel = Arch.nodes.find(n => n.kind === 'router' && n.li === rows[i].li) || Arch.sel;
        Osc.setSubview('arch');
        const b = Arch.bands[rows[i].li];
        if (b) { Arch.cam.tz = Math.max(Arch.cam.tz, 0.85); Arch.cam.ty = -b.y0 * Arch.cam.tz + 30; Arch.syncZoomUI(); }
        Arch._dirty = true; Arch.renderInspector();
      });
    });
  },

  renderDetail(cells) {
    const card = document.getElementById('moe-detail-card');
    const box = document.getElementById('moe-detail');
    if (!this.sel) { card.hidden = true; return; }
    card.hidden = false;
    const c = cells[this.sel.li][this.sel.e];
    box.innerHTML = '';
    box.append(
      UI.el('div', { class: 'pname', style: { color: '#f472b6' } }, `L${c.li} · expert ${c.e}`),
      UI.el('div', { class: 'kv', style: { marginTop: '8px' } },
        UI.el('span', { class: 'k' }, '当前 step 负载'), UI.el('span', { class: 'v' }, (c.shares[Osc.step] * 100).toFixed(2) + '%'),
        UI.el('span', { class: 'k' }, '当前 |g|'), UI.el('span', { class: 'v' }, UI.num(c.gs[Osc.step], 5)),
        UI.el('span', { class: 'k' }, '窗口平均负载'), UI.el('span', { class: 'v' }, (c.meanShare * 100).toFixed(2) + '%'),
        UI.el('span', { class: 'k' }, '路由频率'), UI.el('span', { class: 'v' }, (c.freq * 100).toFixed(1) + '%'),
        UI.el('span', { class: 'k' }, '最长连续未路由'), UI.el('span', { class: 'v' }, c.streak + ' step')));
    const spark = UI.el('canvas', { style: { width: '100%', display: 'block', marginTop: '10px' } });
    box.append(spark);
    const w = box.clientWidth || 400, h = 64;
    const ctx = Osc.fit(spark, w, h);
    ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, w, h);
    const mx = Math.max(1e-9, ...c.shares);
    const bw = (w - 8) / Math.max(1, c.shares.length);
    c.shares.forEach((v, i) => {
      const bh = Math.max(v > 0 ? 2 : 0, (v / mx) * (h - 14));
      ctx.fillStyle = c.act[i] ? Osc.ramp(v / mx) : '#3a1220';
      ctx.fillRect(4 + i * bw, h - 6 - bh, Math.max(1, bw - 1.5), bh);
    });
    ctx.fillStyle = '#e8eef4';
    ctx.fillRect(4 + Osc.step * bw + bw / 2 - 0.5, 2, 1, h - 4);
    spark.addEventListener('click', ev => {
      const r = spark.getBoundingClientRect();
      Osc.setStep(Math.floor((ev.clientX - r.left - 4) / bw));
    });
    box.append(UI.el('div', { style: { marginTop: '10px', display: 'flex', gap: '8px' } },
      UI.el('button', {
        class: 'btn sm', onclick: () => {
          Arch.sel = Arch.nodes.find(n => n.kind === 'expert' && n.li === c.li && n.e === c.e) || Arch.sel;
          Osc.setSubview('arch'); Arch._dirty = true; Arch.renderInspector();
        },
      }, '在架构图中定位'),
      UI.el('button', { class: 'btn sm ghost', onclick: () => { this.sel = null; this.render(); } }, '关闭')));
  },

  renderLegend(box) {
    const names = { load: '负载代理占比', active: '路由激活', grad: '专家梯度 L2' };
    box.append(
      UI.el('span', {}, names[this.metric]),
      UI.el('span', { class: 'bar', style: { background: Osc.rampCSS() } }),
      UI.el('span', { class: 'lg-zero' }, UI.el('i', {}), '未路由'),
      UI.el('span', { class: 'lg-hungry' }, UI.el('i', {}), `饥饿 <${Arch.hungry}%`));
  },

  exportCSV() {
    const { cells, S, E, T } = this.compute();
    const lines = ['layer,expert,routed_steps,steps,routing_freq,mean_load_proxy,max_load_proxy,longest_unrouted_streak,mean_grad_l2'];
    for (let li = 0; li < S; li++) for (let e = 0; e < E; e++) {
      const c = cells[li][e];
      lines.push([li, e, c.na, T, c.freq.toFixed(4), c.meanShare.toFixed(5), c.maxShare.toFixed(5), c.streak, c.meanG.toExponential(4)].join(','));
    }
    UI.download(`moe-routing-${App.currentJob}.csv`, lines.join('\n'), 'text/csv');
    UI.toast('MoE 路由统计已导出 CSV');
  },

  bind() {
    const $ = id => document.getElementById(id);
    $('moe-metric').addEventListener('change', e => { this.metric = e.target.value; this.render(); Osc.renderLegend(); });
    $('moe-window').addEventListener('change', e => { this.window = e.target.value; this.render(); });
    $('moe-csv').addEventListener('click', () => this.exportCSV());
    const canvas = $('moe-matrix');
    const tip = this._tip = UI.el('div', { class: 'ctip' });
    document.body.append(tip);
    const hit = ev => {
      const r = canvas.getBoundingClientRect();
      const x = ev.clientX - r.left, y = ev.clientY - r.top;
      return (this._mrects || []).find(q => x >= q.x && x <= q.x + q.w && y >= q.y && y <= q.y + q.h);
    };
    canvas.addEventListener('mousemove', ev => {
      const q = hit(ev);
      if (!q) { tip.style.display = 'none'; canvas.style.cursor = 'default'; return; }
      canvas.style.cursor = 'pointer';
      const c = q.c;
      tip.innerHTML = `<b>L${c.li} · expert ${c.e}</b><span class="t2">step ${Osc.steps[Osc.step]} · ${c.act[Osc.step] ? '负载 ' + (c.shares[Osc.step] * 100).toFixed(2) + '% · |g| ' + UI.num(c.gs[Osc.step], 4) : '未路由 ZERO'} · 窗口路由率 ${(c.freq * 100).toFixed(0)}%</span>`;
      tip.style.display = 'block';
      tip.style.left = Math.min(window.innerWidth - 320, ev.clientX + 14) + 'px';
      tip.style.top = (ev.clientY + 14) + 'px';
    });
    canvas.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
    canvas.addEventListener('click', ev => {
      const q = hit(ev);
      if (!q) return;
      this.sel = { li: q.c.li, e: q.c.e };
      this.render();
    });
    window.addEventListener('resize', UI.debounce(() => { if (this.visible) this.render(); }, 150));
    Osc.sub(kind => {
      if (kind === 'view') {
        this.visible = !document.getElementById('grad-sub-moe').hidden;
        if (this.visible) this.render();
      } else if (kind === 'data' || kind === 'rebuild') { this.cache = null; if (this.visible) this.render(); }
      else if (kind === 'step') { if (this.visible) this.render(); }
    });
  },
};