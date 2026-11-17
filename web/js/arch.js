// Architecture dataflow: the real MetaModel forward graph as steerable nodes.
// emb -> [RT gate -> group router -> segmented flash-attn -> MoE top-K] x S
// block outputs b_li feed a Birkhoff/Sinkhorn doubly-stochastic mixing bus;
// each readout row R_li mixes b_0..b_li into the next layer input; tied head.
const Arch = {
  meta: { S: 0, E: 0, G: null, K: null, D: null, H: null },
  nodes: [], edges: [], strands: [], bands: [], world: { w: 1080, h: 800 },
  cam: { x: 0, y: 0, z: 0.8, tx: 0, ty: 0, tz: 0.8 },
  metric: 'l2', flow: 'both', speed: 1, cell: 'load', pow: 2, hungry: 4,
  experts: true, labels: true, bus: true, focus: 'all',
  sel: null, hover: null, visible: false,
  _dirty: true, _raf: 0, _last: 0, _parts: [], _static: null, _drag: null, _mini: null,

  KIND: {
    emb: { c: '#60a5fa', t: 'Token Embedding' },
    rt: { c: '#34d399', t: 'RT 门控混合' },
    route: { c: '#a3e635', t: '组路由 argmax' },
    attn: { c: '#2dd4bf', t: '分段 Flash-Attn' },
    router: { c: '#a78bfa', t: 'MoE 路由 top-K' },
    expert: { c: '#f472b6', t: 'Expert' },
    merge: { c: '#fbbf24', t: 'Sinkhorn 读出行' },
    head: { c: '#fb7185', t: 'RMSNorm + tied logits' },
    other: { c: '#64748b', t: '其他参数' },
  },
  layerHue(li) { return `hsl(${(165 + li * 27) % 360} 72% 62%)`; },

  setMeta(job) {
    const p = (job && job.spec && job.spec.params) || {};
    let S = 0, E = 0;
    for (const o of Osc.order) {
      const m = o.module.match(/^L(\d+)\./);
      if (m) S = Math.max(S, +m[1] + 1);
      const e = o.name.match(/\.moe\.experts\.(\d+)\./);
      if (e) E = Math.max(E, +e[1] + 1);
    }
    this.meta = {
      S, E,
      G: p.n_groups ?? null, K: p.top_k ?? null, D: p.d_model ?? null, H: p.n_heads ?? null,
    };
    const fs = document.getElementById('arch-focus');
    const cur = this.focus;
    fs.innerHTML = '';
    fs.append(UI.el('option', { value: 'all' }, '全部层'));
    for (let i = 0; i < S; i++) fs.append(UI.el('option', { value: String(i) }, '聚焦 L' + i));
    fs.value = [...fs.options].some(o => o.value === cur) ? cur : 'all';
    this.focus = fs.value;
    document.getElementById('grad-modelinfo').textContent =
      `S=${S} · E=${E} · K=${this.meta.K ?? '—'} · G=${this.meta.G ?? '—'} · d=${this.meta.D ?? '—'}`;
    this.build();
  },

  weightsOf(pred) { return Osc.order.filter(pred).map(o => o.name); },

  build() {
    const { S, E } = this.meta;
    this.nodes = []; this.edges = []; this.strands = []; this.bands = [];
    if (!S || !Osc.steps.length) { this.world = { w: 1080, h: 600 }; this.sel = null; this.hover = null; this._dirty = true; return; }
    const BH0 = this.experts ? 0 : 152;
    const TOP = 16, X = { merge: 100, rt: 236, route: 402, attn: 536, moe: 700 };
    this.TOP = TOP;
    const W = { merge: 118, rt: 150, route: 118, attn: 150, moe: 316 };
    const RET = 1046, BUSR = 64;
    this.RET = RET;
    const ecols = E <= 8 ? 4 : E <= 32 ? 8 : 12;
    const erows = Math.max(1, Math.ceil(E / ecols));
    const MH = this.experts ? 46 + erows * 30 + (erows - 1) * 4 + 26 : 96;
    const BH = Math.max(BH0, MH + 42);
    const add = n => { this.nodes.push(n); return n; };
    const emb = add({ id: 'emb', kind: 'emb', li: -1, x: X.merge, y: TOP, w: W.merge, h: 54, title: 'emb.weight', sub: 'vocab × d · tied', weights: this.weightsOf(o => o.module === 'emb') });
    let y = TOP + 70;
    let prevOut = emb;
    for (let li = 0; li < S; li++) {
      const y0 = y;
      const P = `decoder_layers.${li}.`;
      const band = { li, y0, h: BH };
      this.bands.push(band);
      let input;
      if (li === 0) {
        input = add({ id: 'pass0', kind: 'merge', li, pass: true, x: X.merge, y: y0 + 40, w: W.merge, h: 40, title: 'b₀ 直传', sub: 'layer 0 无合并参数', weights: [] });
      } else {
        input = add({
          id: 'merge.' + (li - 1), kind: 'merge', li, fan: li, x: X.merge, y: y0 + 34, w: W.merge, h: 58,
          title: `R${li - 1} 读出行`, sub: `${li}×${li} 双随机凸组合`, weights: this.weightsOf(o => o.name === `merge.res.${li - 1}`),
        });
      }
      const rt = add({
        id: `L${li}.rt`, kind: 'rt', li, x: X.rt, y: y0 + 16, w: W.rt, h: 104,
        title: 'RT 门控', sub: 'σ(w₁x)·w₂x + rs·res', weights: this.weightsOf(o => o.module === `L${li}.rt` && !o.name.includes('group_gate')),
      });
      const route = add({
        id: `L${li}.route`, kind: 'route', li, x: X.route, y: y0 + 16, w: W.route, h: 104,
        title: '组路由', sub: `G=${this.meta.G ?? '?'} · softmax→argmax`, weights: this.weightsOf(o => o.module === `L${li}.rt` && o.name.includes('group_gate')),
      });
      const attn = add({
        id: `L${li}.attn`, kind: 'attn', li, x: X.attn, y: y0 + 16, w: W.attn, h: 104,
        title: 'Attention', sub: `qkv+RoPE · H=${this.meta.H ?? '?'} 段内 causal`, weights: this.weightsOf(o => o.module === `L${li}.attn`),
      });
      const mh = MH;
      const router = add({
        id: `L${li}.moe`, kind: 'router', li, x: X.moe, y: y0 + 8, w: W.moe, h: mh,
        title: 'MoE gate', sub: `E=${E} · K=${this.meta.K ?? '?'} · softmax→topK`, weights: this.weightsOf(o => o.module === `L${li}.moe` && !o.name.includes('.experts.')),
      });
      if (this.experts && E) {
        const cols = ecols;
        const rows = erows;
        const gx = router.x + 10, gy = router.y + 46;
        const gw = router.w - 20, gh = router.h - 46 - 26;
        const cw = (gw - (cols - 1) * 4) / cols, ch = Math.min(30, (gh - (rows - 1) * 4) / rows);
        for (let e = 0; e < E; e++) {
          const cx = gx + (e % cols) * (cw + 4), cy = gy + Math.floor(e / cols) * (ch + 4);
          add({
            id: `L${li}.e${e}`, kind: 'expert', li, e, x: cx, y: cy, w: cw, h: ch,
            title: `expert ${e}`, sub: 'gate_up → down', weights: this.weightsOf(o => o.name === `${P}moe.experts.${e}.down.weight` || o.name === `${P}moe.experts.${e}.gate_up.weight`),
          });
        }
      }
      const tapY = y0 + BH - 16;
      band.tapY = tapY; band.router = router; band.input = input;
      // main forward path
      this.link(prevOut, input, 'main');
      this.link(input, rt, 'main');
      this.link(rt, route, 'main');
      this.link(route, attn, 'main');
      this.link(attn, router, 'main');
      const sx = BUSR - 6 - li * 4;
      this.strands.push({ li, x: sx, y0: tapY, y1: 0, hue: this.layerHue(li) });
      this.edges.push({
        kind: 'ret', li, mag: 1, pts: [
          { x: router.x + router.w, y: router.y + router.h / 2 }, { x: RET, y: router.y + router.h / 2 },
          { x: RET, y: tapY }, { x: sx, y: tapY }],
      });
      prevOut = router;
      y += BH;
    }
    const yEnd = y;
    const out = add({
      id: 'merge.out', kind: 'merge', li: S - 1, fan: S, out: true, x: X.merge, y: yEnd + 18, w: W.merge, h: 58,
      title: `R${S - 1} 最终读出`, sub: `${S}×${S} 双随机凸组合`, weights: this.weightsOf(o => o.name === `merge.res.${S - 1}`),
    });
    const head = add({
      id: 'head', kind: 'head', li: -1, x: X.rt, y: yEnd + 18, w: W.rt, h: 58,
      title: 'head_norm', sub: 'logits = norm(h)·embᵀ', weights: this.weightsOf(o => o.module === 'head'),
    });
    const other = this.weightsOf(o => o.module === 'other');
    if (other.length) add({ id: 'other', kind: 'other', li: -1, x: X.route, y: yEnd + 18, w: W.route + W.attn + 16, h: 58, title: '其他参数', sub: other.length + ' 个未归类权重', weights: other });
    this.link(out, head, 'main');
    for (const st of this.strands) st.y1 = out.y + out.h / 2;
    for (const n of this.nodes) {
      if (n.kind !== 'merge' || n.pass) continue;
      const cy = n.y + n.h / 2;
      const upto = n.out ? S - 1 : n.fan - 1;
      for (let j = 0; j <= upto; j++)
        this.edges.push({ kind: 'fan', li: n.li, j, node: n.id, mag: 0.4, pts: [{ x: this.strands[j].x, y: cy }, { x: n.x, y: cy }] });
    }
    this.edges.push({
      kind: 'tied', mag: 0.5, pts: [
        { x: head.x + head.w, y: head.y + head.h / 2 }, { x: RET + 18, y: head.y + head.h / 2 },
        { x: RET + 18, y: TOP + 27 }, { x: emb.x + emb.w, y: TOP + 27 }],
    });
    for (const e of this.edges) this.measure(e);
    this.world = { w: RET + 40, h: yEnd + 108 };
    this.sel = this.sel && this.nodes.find(n => n.id === this.sel.id) ? this.nodes.find(n => n.id === this.sel.id) : this.nodes.find(n => n.kind === 'router' && n.li === S - 1) || null;
    this.seedParticles();
    this._dirty = true;
  },

  link(a, b, kind) {
    const p0 = { x: a.x + a.w, y: a.y + a.h / 2 }, p2 = { x: b.x, y: b.y + b.h / 2 };
    const mx = (p0.x + p2.x) / 2;
    const pts = [];
    for (let i = 0; i <= 10; i++) {
      const t = i / 10, u = 1 - t;
      pts.push({ x: u * u * p0.x + 2 * u * t * mx + t * t * p2.x, y: u * u * p0.y + 2 * u * t * ((p0.y + p2.y) / 2) + t * t * p2.y });
    }
    this.edges.push({ kind, from: a.id, to: b.id, li: b.li, mag: 1, pts });
  },
  measure(e) {
    e.cum = [0];
    for (let i = 1; i < e.pts.length; i++)
      e.cum.push(e.cum[i - 1] + Math.hypot(e.pts[i].x - e.pts[i - 1].x, e.pts[i].y - e.pts[i - 1].y));
    e.len = e.cum[e.cum.length - 1] || 1;
  },
  pointAt(e, t) {
    const d = t * e.len;
    let i = 1;
    while (i < e.cum.length - 1 && e.cum[i] < d) i++;
    const f = (d - e.cum[i - 1]) / ((e.cum[i] - e.cum[i - 1]) || 1);
    return { x: e.pts[i - 1].x + (e.pts[i].x - e.pts[i - 1].x) * f, y: e.pts[i - 1].y + (e.pts[i].y - e.pts[i - 1].y) * f };
  },

  // ---- per-step metrics -------------------------------------------------
  stats() {
    const vec = Osc.vecAt(Osc.step);
    const g = n => { let s = 0, z = 0, c = 0; for (const w of n.weights) { const v = Osc.l2(w); if (v === null) continue; c++; s += v * v; if (v === 0) z++; } return { l2: Math.sqrt(s), e: s, z, c }; };
    let total = 0;
    for (const v of vec) if (v > 0) total += v * v;
    const byId = new Map();
    for (const n of this.nodes) {
      const s = g(n);
      n.l2 = s.l2; n.energy = s.e; n.zero = s.z; n.nw = s.c;
      n.share = total > 0 ? s.e / total : 0;
      byId.set(n.id, n);
    }
    this.total = total;
    this.layerE = [];
    for (let li = 0; li < this.meta.S; li++) {
      let e = 0;
      for (const n of this.nodes) if (n.li === li) e += n.energy;
      this.layerE.push(e);
    }
    // MoE per-layer expert load proxy from gradient energy (zero grad = unrouted)
    this.moe = [];
    for (let li = 0; li < this.meta.S; li++) {
      const gs = [];
      for (let e = 0; e < this.meta.E; e++) {
        const a = Osc.l2(`decoder_layers.${li}.moe.experts.${e}.down.weight`);
        const b = Osc.l2(`decoder_layers.${li}.moe.experts.${e}.gate_up.weight`);
        gs.push(a === null && b === null ? 0 : Math.sqrt((a || 0) ** 2 + (b || 0) ** 2));
      }
      const pw = gs.map(v => Math.pow(v, this.pow));
      const sum = pw.reduce((a, b) => a + b, 0);
      const shares = pw.map(v => sum > 0 ? v / sum : 0);
      const act = gs.map(v => v > 0);
      const na = act.filter(Boolean).length;
      let H = 0;
      for (const s of shares) if (s > 0) H -= s * Math.log(s);
      const E = Math.max(1, this.meta.E);
      const mean = 1 / E;
      const cv = Math.sqrt(shares.reduce((a, s) => a + (s - mean) * (s - mean), 0) / E) / mean;
      const mx = Math.max(0, ...shares);
      for (const n of this.nodes) if (n.kind === 'expert' && n.li === li) {
        n.load = shares[n.e]; n.active = act[n.e]; n.g = gs[n.e];
        n.hungry = act[n.e] && shares[n.e] * 100 < this.hungry;
      }
      const r = byId.get(`L${li}.moe`);
      if (r) { r.activeCount = na; r.expN = this.meta.E; r.H = H; r.eff = Math.exp(H); r.cv = cv; r.maxShare = mx; }
      this.moe.push({ li, shares, act, na, n: this.meta.E, H, eff: Math.exp(H), nH: E > 1 ? H / Math.log(E) : 1, cv, maxShare: mx, maxLoad: mx });
    }
    this.maxShare = Math.max(1e-9, ...this.nodes.map(n => n.share));
    return byId;
  },
  nodeT(n) {
    switch (this.metric) {
      case 'share': return Math.sqrt(n.share / this.maxShare);
      case 'zero': return n.nw ? n.zero / n.nw : 0;
      case 'active':
        if (n.kind === 'expert') return n.active ? 0.9 : 0.04;
        if (n.kind === 'router') return n.expN ? n.activeCount / n.expN : 0;
        return n.nw ? 1 - n.zero / n.nw : 0;
      case 'load':
        if (n.kind === 'expert') { const m = this.moe[n.li]; return m && m.maxLoad > 0 ? n.load / m.maxLoad : 0; }
        if (n.kind === 'router') return n.expN ? n.activeCount / n.expN : 0;
        return Math.sqrt(n.share / this.maxShare);
      default: return Osc.norm(n.l2) ?? 0;
    }
  },

  seedParticles() {
    this._parts = [];
    for (const e of this.edges) {
      if (e.kind === 'fan') continue;
      const n = e.kind === 'ret' ? 3 : e.kind === 'tied' ? 2 : e.kind === 'main' ? 3 : 2;
      for (let i = 0; i < n; i++) this._parts.push({ e, t: Math.random(), dir: 1 });
      if (e.kind !== 'tied') for (let i = 0; i < n; i++) this._parts.push({ e, t: Math.random(), dir: -1 });
    }
  },

  // ---- rendering --------------------------------------------------------
  render() {
    this.visible = !document.getElementById('grad-sub-arch').hidden;
    if (!this.visible) return;
    const wrap = document.getElementById('arch-wrap');
    if (!wrap.clientWidth) return;
    if (!this.nodes.length) this.build();
    this.stats();
    this.resize();
    if (!this._fitted && this.meta.S) { this.fitWidth(); this._fitted = true; }
    this.cam.x = this.cam.tx; this.cam.y = this.cam.ty; this.cam.z = this.cam.tz;
    this.renderStatic(); this.renderMini();
    this.renderInspector();
    this.renderStep();
    this._dirty = true;
    this.start();
  },
  resize() {
    const wrap = document.getElementById('arch-wrap');
    const em = document.getElementById('arch-empty');
    const mini = document.getElementById('arch-mini');
    const c = document.getElementById('arch-canvas');
    const has = !!(Osc.steps.length && this.meta.S);
    em.hidden = has;
    c.style.display = has ? 'block' : 'none';
    mini.style.display = has ? 'block' : 'none';
    const h = wrap.clientHeight;
    const md = Osc.dpr();
    // size the docked minimap first: it consumes stage width, so the canvas
    // must be measured after the minimap has its final box
    mini.width = 56 * md; mini.height = h * md;
    mini.style.width = '56px'; mini.style.height = h + 'px';
    const w = wrap.clientWidth;
    const d = Osc.dpr();
    if (c.width !== Math.round(w * d) || c.height !== Math.round(h * d)) {
      c.width = Math.round(w * d); c.height = Math.round(h * d);
      c.style.width = w + 'px'; c.style.height = h + 'px';
      this._static = document.createElement('canvas');
      this._static.width = c.width; this._static.height = c.height;
      this._dirty = true;
    }
    this.vw = w; this.vh = h;
  },
  fitWidth() {
    const z = Math.min(1.15, Math.max(0.5, (this.vw - 20) / this.world.w));
    this.cam.tz = z; this.cam.tx = Math.min(10, (this.vw - this.world.w * z) / 2); this.cam.ty = 8;
    this.syncZoomUI();
  },
  fitAll() {
    const z = Math.min((this.vw - 16) / this.world.w, (this.vh - 16) / this.world.h);
    this.cam.tz = Math.max(0.12, z);
    this.cam.tx = (this.vw - this.world.w * this.cam.tz) / 2;
    this.cam.ty = (this.vh - this.world.h * this.cam.tz) / 2;
    this.syncZoomUI();
  },
  syncZoomUI() {
    const s = document.getElementById('arch-zoom');
    s.value = String(Math.round(this.cam.tz * 100));
    document.getElementById('arch-zoom-label').textContent = Math.round(this.cam.tz * 100) + '%';
  },
  start() {
    if (this._raf) return;
    this._last = performance.now();
    const loop = t => {
      this._raf = requestAnimationFrame(loop);
      const dt = Math.min(0.05, (t - this._last) / 1000); this._last = t;
      if (!this.visible) return;
      const c = this.cam, k = Math.min(1, dt * 12);
      const moved = Math.abs(c.tx - c.x) > 0.1 || Math.abs(c.ty - c.y) > 0.1 || Math.abs(c.tz - c.z) > 0.001;
      if (moved) { c.x += (c.tx - c.x) * k; c.y += (c.ty - c.y) * k; c.z += (c.tz - c.z) * k; this._dirty = true; }
      if (this._dirty) { this.renderStatic(); this.renderMini(); this._dirty = false; }
      const ctx = document.getElementById('arch-canvas').getContext('2d');
      const d = Osc.dpr();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.drawImage(this._static, 0, 0);
      if (this.flow !== 'off' && Osc.steps.length) this.drawParticles(ctx, d, dt);
    };
    this._raf = requestAnimationFrame(loop);
  },
  stop() { cancelAnimationFrame(this._raf); this._raf = 0; },

  renderStatic() {
    const ctx = this._static.getContext('2d');
    const d = Osc.dpr();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, this._static.width, this._static.height);
    this.scene(ctx, d, this.cam);
  },
  scene(ctx, d, cam) {
    ctx.setTransform(d * cam.z, 0, 0, d * cam.z, d * cam.x, d * cam.y);
    // ambient grid
    ctx.strokeStyle = 'rgba(38,49,62,.35)'; ctx.lineWidth = 1 / cam.z;
    const g = 64;
    const vw = ctx.canvas.width / (d * cam.z), vh = ctx.canvas.height / (d * cam.z);
    const x0 = Math.floor(-cam.x / cam.z / g) * g, x1 = x0 + vw + g;
    const y0 = Math.floor(-cam.y / cam.z / g) * g, y1 = y0 + vh + g;
    ctx.beginPath();
    for (let x = x0; x < x1; x += g) { ctx.moveTo(x, y0); ctx.lineTo(x, y1); }
    for (let y = y0; y < y1; y += g) { ctx.moveTo(x0, y); ctx.lineTo(x1, y); }
    ctx.stroke();
    const selId = this.sel && this.sel.id, hovId = this.hover && this.hover.id;
    const dim = li => this.focus !== 'all' && li >= 0 && String(li) !== this.focus;
    // mixing bus ribbon
    if (this.bus && this.strands.length) {
      const rl = this.strands[this.strands.length - 1].x - 6;
      ctx.fillStyle = 'rgba(23,31,41,.72)';
      ctx.fillRect(rl, this.bands[0].tapY - 8, this.strands[0].x - rl + 10, this.strands[0].y1 - this.bands[0].tapY + 16);
      ctx.font = '600 9.5px ui-monospace, Consolas, monospace';
      ctx.save(); ctx.translate(rl - 6, (this.bands[0].tapY + this.strands[0].y1) / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = '#4b5a6b'; ctx.textAlign = 'center'; ctx.fillText('BIRKHOFF 混合总线 · b₀…b' + (this.meta.S - 1), 0, 0); ctx.restore();
      ctx.textAlign = 'left';
      for (const st of this.strands) {
        const hot = this.sel && this.sel.kind === 'merge' && !this.sel.pass && st.li <= (this.sel.out ? this.meta.S - 1 : this.sel.fan - 1);
        ctx.globalAlpha = dim(st.li) ? 0.12 : hot ? 0.95 : 0.4;
        ctx.strokeStyle = st.hue; ctx.lineWidth = hot ? 2.4 : 1.6;
        ctx.beginPath(); ctx.moveTo(st.x, st.y0); ctx.lineTo(st.x, st.y1); ctx.stroke();
        ctx.fillStyle = st.hue; ctx.beginPath(); ctx.arc(st.x, st.y0, hot ? 3.4 : 2.4, 0, 7); ctx.fill();
        ctx.globalAlpha = 1;
      }
    }
    // edges
    for (const e of this.edges) {
      const li = e.li ?? -1;
      const hotFan = e.kind === 'fan' && selId === e.node;
      const hotRet = e.kind === 'ret' && this.sel && this.sel.li === e.li && (this.sel.kind === 'expert' || this.sel.kind === 'router');
      ctx.globalAlpha = dim(li) ? 0.1 : e.kind === 'fan' ? (hotFan ? 0.95 : 0.22) : e.kind === 'tied' ? 0.4 : hotRet ? 0.95 : 0.55;
      ctx.strokeStyle = e.kind === 'fan' ? this.layerHue(e.j) : e.kind === 'ret' ? this.layerHue(e.li) : e.kind === 'tied' ? '#60a5fa' : '#3d4c5e';
      ctx.lineWidth = (hotFan || hotRet ? 2 : e.kind === 'main' ? 1.6 : 1.2) / Math.max(0.6, cam.z * 0.9);
      if (e.kind === 'tied') ctx.setLineDash([5, 4]);
      ctx.beginPath();
      e.pts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
      ctx.stroke(); ctx.setLineDash([]);
      if (e.kind === 'main' || e.kind === 'ret') {
        const p = e.pts[e.pts.length - 1], q = e.pts[e.pts.length - 2];
        const a = Math.atan2(p.y - q.y, p.x - q.x);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath(); ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x - 6 * Math.cos(a - 0.4), p.y - 6 * Math.sin(a - 0.4));
        ctx.lineTo(p.x - 6 * Math.cos(a + 0.4), p.y - 6 * Math.sin(a + 0.4));
        ctx.closePath(); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    // band headers
    ctx.textBaseline = 'middle';
    for (const b of this.bands) {
      ctx.globalAlpha = dim(b.li) ? 0.25 : 1;
      ctx.font = '700 13px ui-monospace, Consolas, monospace';
      ctx.fillStyle = this.layerHue(b.li);
      ctx.fillText('L' + b.li, 100, b.y0 + 8);
      ctx.font = '9.5px ui-monospace, Consolas, monospace';
      ctx.fillStyle = '#4b5a6b';
      const le = this.layerE[b.li] || 0;
      ctx.fillText(`decoder layer · 能量 ${this.total > 0 ? (le / this.total * 100).toFixed(1) : '0.0'}%`, 126, b.y0 + 8);
      ctx.globalAlpha = 1;
    }
    // nodes
    for (const n of this.nodes) {
      if (n.kind === 'expert') { this.drawExpert(ctx, n, cam, dim); continue; }
      this.drawNode(ctx, n, cam, dim, selId, hovId);
    }
    // logits caption
    const head = this.nodes.find(n => n.id === 'head');
    if (head) {
      ctx.font = '10px ui-monospace, Consolas, monospace';
      ctx.fillStyle = '#7d8fa3';
      ctx.fillText('→ logits (V)', head.x + head.w + 12, head.y + head.h / 2);
      ctx.fillStyle = '#4b5a6b';
      ctx.fillText('tied embᵀ', this.RET - 62, this.TOP + 14);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  },
  drawNode(ctx, n, cam, dim, selId, hovId) {
    const K = this.KIND[n.kind];
    const a = dim(n.li) ? 0.22 : 1;
    ctx.globalAlpha = a;
    const t = this.nodeT(n);
    ctx.fillStyle = '#121820';
    this.rr(ctx, n.x, n.y, n.w, n.h, 8); ctx.fill();
    ctx.fillStyle = Osc.ramp(t);
    ctx.globalAlpha = a * (0.05 + 0.11 * t);
    this.rr(ctx, n.x, n.y, n.w, n.h, 8); ctx.fill();
    ctx.globalAlpha = a;
    ctx.strokeStyle = n.id === selId ? '#e8eef4' : n.id === hovId ? K.c : 'rgba(61,76,94,.9)';
    ctx.lineWidth = n.id === selId ? 1.8 : 1.1;
    this.rr(ctx, n.x, n.y, n.w, n.h, 8); ctx.stroke();
    ctx.fillStyle = K.c; ctx.fillRect(n.x, n.y + 6, 3, n.h - 12);
    // heat chip: current metric intensity at a glance
    ctx.fillStyle = Osc.ramp(t);
    this.rr(ctx, n.x + n.w - 17, n.y + 8, 9, 9, 2); ctx.fill();
    ctx.strokeStyle = 'rgba(10,14,20,.8)'; ctx.lineWidth = 1;
    this.rr(ctx, n.x + n.w - 17, n.y + 8, 9, 9, 2); ctx.stroke();
    ctx.textBaseline = 'middle';
    ctx.font = '600 13px ui-monospace, Consolas, monospace';
    ctx.fillStyle = '#e8eef4';
    ctx.fillText(n.title, n.x + 11, n.y + 15);
    ctx.font = '10px ui-monospace, Consolas, monospace';
    ctx.fillStyle = '#7d8fa3';
    const maxc = Math.floor((n.w - 34) / 5.7);
    const sub = n.sub.length > maxc ? n.sub.slice(0, maxc - 1) + '…' : n.sub;
    ctx.fillText(sub, n.x + 11, n.y + 29);
    ctx.font = '11px ui-monospace, Consolas, monospace';
    if (n.kind === 'router') {
      const m = this.moe[n.li];
      ctx.fillStyle = '#a78bfa';
      ctx.fillText(`激活 ${m ? m.na : 0}/${m ? m.n : 0} · eff ${m ? m.eff.toFixed(1) : '—'}`, n.x + 11, n.y + 41);
      if (!this.experts) {
        const bw = n.w - 22;
        ctx.fillStyle = '#1c2530'; ctx.fillRect(n.x + 11, n.y + 56, bw, 10);
        if (m) {
          let x = n.x + 11;
          m.shares.forEach((s, i) => { ctx.fillStyle = m.act[i] ? Osc.ramp(s / (m.maxLoad || 1)) : '#3a1220'; ctx.fillRect(x, n.y + 56, Math.max(0.5, bw * s), 10); x += bw * s; });
        }
      }
      ctx.fillStyle = '#7d8fa3';
      ctx.fillText(`CV ${m ? m.cv.toFixed(2) : '—'} · 熵 ${m ? m.nH.toFixed(2) : '—'} · max ${m ? (m.maxShare * m.n).toFixed(1) : '—'}×`, n.x + 11, n.y + n.h - 12);
    } else if (n.kind === 'merge' && !n.pass) {
      ctx.fillStyle = '#fbbf24';
      ctx.fillText(`fan-in ${n.out ? this.meta.S : n.fan} · |g| ${UI.num(n.l2, 3)}`, n.x + 11, n.y + 44);
    } else if (n.pass) {
      ctx.fillStyle = '#4b5a6b'; ctx.fillText('x₀ = emb', n.x + 11, n.y + 30);
    } else {
      ctx.fillStyle = K.c;
      ctx.fillText(`|g| ${UI.num(n.l2, 4)}`, n.x + 11, n.y + 46);
      ctx.fillStyle = '#7d8fa3';
      ctx.fillText(`能量 ${(n.share * 100).toFixed(1)}%`, n.x + 11, n.y + 60);
      if (n.zero) { ctx.fillStyle = '#f87171'; ctx.fillText(`零梯度 ×${n.zero}`, n.x + 11, n.y + 74); }
      if (this.labels && n.h >= 100) {
        const bw = n.w - 22;
        ctx.fillStyle = '#1c2530'; ctx.fillRect(n.x + 11, n.y + n.h - 16, bw, 5);
        ctx.fillStyle = K.c; ctx.fillRect(n.x + 11, n.y + n.h - 16, bw * Math.min(1, n.share / this.maxShare), 5);
      }
    }
    if (n.zero && n.nw && n.zero === n.nw && n.kind !== 'expert') {
      ctx.strokeStyle = '#f87171'; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      this.rr(ctx, n.x + 2, n.y + 2, n.w - 4, n.h - 4, 6); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.globalAlpha = 1;
  },
  drawExpert(ctx, n, cam, dim) {
    const a = dim(n.li) ? 0.2 : 1;
    ctx.globalAlpha = a;
    const t = this.cell === 'grad' ? (Osc.norm(n.g) ?? 0) : this.cell === 'active' ? (n.active ? 0.88 : 0.03) : (this.moe[n.li].maxLoad > 0 ? n.load / this.moe[n.li].maxLoad : 0);
    ctx.fillStyle = n.active ? Osc.ramp(t) : '#2a0f18';
    this.rr(ctx, n.x, n.y, n.w, n.h, 4); ctx.fill();
    const sel = this.sel && this.sel.id === n.id, hov = this.hover && this.hover.id === n.id;
    ctx.strokeStyle = sel ? '#e8eef4' : hov ? '#f472b6' : n.hungry ? '#fbbf24' : n.active ? 'rgba(10,14,20,.6)' : '#7f1d2b';
    ctx.lineWidth = sel || hov ? 1.6 : 1;
    this.rr(ctx, n.x, n.y, n.w, n.h, 4); ctx.stroke();
    if (!n.active) {
      ctx.strokeStyle = 'rgba(248,113,113,.75)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(n.x + 3, n.y + 3); ctx.lineTo(n.x + n.w - 3, n.y + n.h - 3); ctx.stroke();
    }
    if (this.labels && n.w >= 20) {
      ctx.font = '600 9.5px ui-monospace, Consolas, monospace';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = n.active && t > 0.55 ? '#0a0e14' : '#cbd5e1';
      ctx.fillText(String(n.e), n.x + 4, n.y + n.h / 2);
      if (n.w >= 40 && n.active) {
        ctx.textAlign = 'right';
        const pc = n.load * 100;
        ctx.fillText(pc < 0.5 ? '<1%' : pc.toFixed(0) + '%', n.x + n.w - 4, n.y + n.h / 2);
        ctx.textAlign = 'left';
      }
    }
    ctx.globalAlpha = 1;
  },
  rr(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  },
  drawParticles(ctx, d, dt) {
    if (this.flow === 'off') return;
    ctx.setTransform(d * this.cam.z, 0, 0, d * this.cam.z, d * this.cam.x, d * this.cam.y);
    const v = 70 * this.speed;
    for (const p of this._parts) {
      if (p.dir > 0 && this.flow === 'bwd') continue;
      if (p.dir < 0 && this.flow === 'fwd') continue;
      const li = p.e.li ?? -1;
      if (this.focus !== 'all' && li >= 0 && String(li) !== this.focus) continue;
      p.t += (dt * v / p.e.len) * p.dir;
      if (p.t > 1) p.t -= 1; if (p.t < 0) p.t += 1;
      const q = this.pointAt(p.e, p.t);
      const col = p.dir > 0 ? '45,212,191' : '251,191,36';
      ctx.fillStyle = `rgba(${col},.16)`; ctx.beginPath(); ctx.arc(q.x, q.y, 3.4, 0, 7); ctx.fill();
      ctx.fillStyle = `rgba(${col},.95)`; ctx.beginPath(); ctx.arc(q.x, q.y, 1.5, 0, 7); ctx.fill();
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  },
  renderMini() {
    const c = document.getElementById('arch-mini');
    const w = 56, h = this.vh;
    const d = Osc.dpr();
    if (c.width !== w * d || c.height !== h * d) { c.width = w * d; c.height = h * d; c.style.width = w + 'px'; c.style.height = h + 'px'; }
    const ctx = c.getContext('2d');
    ctx.setTransform(d, 0, 0, d, 0, 0);
    ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, w, h);
    const sx = (w - 8) / this.world.w, sy = (h - 8) / this.world.h;
    for (const b of this.bands) {
      const e = this.layerE[b.li] || 0;
      ctx.fillStyle = this.layerHue(b.li);
      ctx.globalAlpha = 0.25 + 0.6 * Math.min(1, e / (this.total || 1) * this.meta.S);
      ctx.fillRect(4 + 100 * sx, 4 + b.y0 * sy, Math.max(2, (this.world.w - 140) * sx), Math.max(2, b.h * sy - 1));
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = '#e8eef4'; ctx.lineWidth = 1.2;
    ctx.strokeRect(4 + (-this.cam.x / this.cam.z) * sx, 4 + (-this.cam.y / this.cam.z) * sy,
      (this.vw / this.cam.z) * sx, (this.vh / this.cam.z) * sy);
  },

  // ---- inspector --------------------------------------------------------
  spark(box, series, color) {
    const spark = UI.el('canvas', { style: { width: '100%', display: 'block' } });
    box.append(spark);
    const w = box.clientWidth || 260, h = 52;
    const ctx = Osc.fit(spark, w, h);
    ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, w, h);
    const vals = series.filter(v => v !== null && v > 0);
    if (!vals.length) return spark;
    const lmin = Math.log10(Math.min(...vals)), lmax = Math.log10(Math.max(...vals));
    const span = (lmax - lmin) || 1;
    ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.beginPath();
    let started = false;
    series.forEach((v, i) => {
      if (v === null) return;
      const x = 4 + (i / Math.max(1, series.length - 1)) * (w - 8);
      const y = h - 7 - ((Math.log10(Math.max(v, 1e-12)) - lmin) / span) * (h - 14);
      started ? ctx.lineTo(x, y) : ctx.moveTo(x, y); started = true;
    });
    ctx.stroke();
    const sx = 4 + (Osc.step / Math.max(1, series.length - 1)) * (w - 8);
    ctx.fillStyle = '#e8eef4'; ctx.fillRect(sx - 0.5, 2, 1, h - 4);
    spark.addEventListener('click', ev => {
      const r = spark.getBoundingClientRect();
      Osc.setStep(Math.round(((ev.clientX - r.left - 4) / (r.width - 8)) * (series.length - 1)));
    });
    return spark;
  },
  nodeSeries(n) {
    return Osc.full.map(vec => {
      let s = 0, any = false;
      for (const w of n.weights) { const oi = Osc._idx.get(w); if (oi === undefined) continue; const v = vec[oi]; if (v === null) continue; any = true; s += v * v; }
      return any ? Math.sqrt(s) : null;
    });
  },
  renderInspector() {
    const box = document.getElementById('arch-detail');
    box.innerHTML = '';
    const n = this.sel;
    if (!n) { box.append(UI.el('div', { class: 'empty' }, UI.el('b', {}, '未选中节点'), '点击图中任意节点 / 专家格')); return; }
    const K = this.KIND[n.kind];
    box.append(
      UI.el('div', { class: 'pname', style: { color: K.c } }, n.kind === 'expert' ? `L${n.li} · expert ${n.e}` : n.title),
      UI.el('div', { class: 'kv-chips', style: { margin: '8px 0' } },
        UI.el('span', { class: 'tag', style: { color: K.c } }, K.t),
        n.li >= 0 ? UI.el('span', { class: 'tag' }, 'L' + n.li) : null,
        n.kind === 'expert' ? UI.el('span', { class: 'tag', style: { color: n.active ? 'var(--green)' : 'var(--red)' } }, n.active ? '已路由' : '未路由') : null,
        n.zero ? UI.el('span', { class: 'zero-badge' }, `零梯度 ×${n.zero}`) : null));
    if (n.kind === 'expert') {
      const m = this.moe[n.li];
      const gd = Osc.l2(`decoder_layers.${n.li}.moe.experts.${n.e}.down.weight`);
      const gg = Osc.l2(`decoder_layers.${n.li}.moe.experts.${n.e}.gate_up.weight`);
      const cov = this.coverage(n.li, n.e);
      box.append(UI.el('div', { class: 'kv' },
        UI.el('span', { class: 'k' }, '负载代理占比'), UI.el('span', { class: 'v' }, (n.load * 100).toFixed(2) + '%'),
        UI.el('span', { class: 'k' }, '合成 |g|'), UI.el('span', { class: 'v' }, UI.num(n.g, 5)),
        UI.el('span', { class: 'k' }, 'down / gate_up'), UI.el('span', { class: 'v' }, `${UI.num(gd, 4)} / ${UI.num(gg, 4)}`),
        UI.el('span', { class: 'k' }, '窗口激活率'), UI.el('span', { class: 'v' }, (cov.freq * 100).toFixed(0) + '%'),
        UI.el('span', { class: 'k' }, '最长连续未路由'), UI.el('span', { class: 'v', style: cov.streak > 3 ? { color: 'var(--red)' } : {} }, cov.streak + ' step'),
        UI.el('span', { class: 'k' }, '层内排名'), UI.el('span', { class: 'v' }, '#' + (1 + m.shares.filter(s => s > n.load).length) + ' / ' + m.n)));
      this.spark(box, this.nodeSeries(n), '#f472b6');
      box.append(UI.el('div', { class: 'note' }, '负载为专家梯度能量占比（g^p 归一）代理；零梯度 = 该 step 无 token 路由到该专家。'));
      return;
    }
    if (n.kind === 'router') {
      const m = this.moe[n.li];
      if (!m) { box.append(UI.el('div', { class: 'note' }, '当前数据不含 MoE 专家权重。')); return; }
      box.append(UI.el('div', { class: 'kv' },
        UI.el('span', { class: 'k' }, '激活专家'), UI.el('span', { class: 'v' }, `${m.na} / ${m.n}`),
        UI.el('span', { class: 'k' }, '有效专家数 exp(H)'), UI.el('span', { class: 'v' }, m.eff.toFixed(2)),
        UI.el('span', { class: 'k' }, '归一化熵'), UI.el('span', { class: 'v' }, m.nH.toFixed(3)),
        UI.el('span', { class: 'k' }, '负载 CV'), UI.el('span', { class: 'v', style: m.cv > 1 ? { color: 'var(--amber)' } : {} }, m.cv.toFixed(3)),
        UI.el('span', { class: 'k' }, 'max / mean'), UI.el('span', { class: 'v' }, (m.maxShare * m.n).toFixed(2)),
        UI.el('span', { class: 'k' }, 'gate |g|'), UI.el('span', { class: 'v' }, UI.num(n.l2, 5))));
      const list = UI.el('div', { class: 'exlist' });
      this.nodes.filter(q => q.kind === 'expert' && q.li === n.li).forEach(q => {
        list.append(UI.el('button', {
          class: 'exrow' + (q.active ? '' : ' dead') + (q.hungry ? ' hungry' : ''),
          onclick: () => { this.sel = q; this._dirty = true; this.renderInspector(); },
        },
          UI.el('span', { class: 'i' }, 'e' + q.e),
          UI.el('span', { class: 'b' }, UI.el('i', { style: { width: (q.load * 100).toFixed(1) + '%', background: q.active ? Osc.ramp(q.load / (m.maxLoad || 1)) : 'var(--red)' } })),
          UI.el('span', { class: 'p' }, q.active ? (q.load * 100).toFixed(1) + '%' : 'dead')));
      });
      box.append(list);
      this.spark(box, this.nodeSeries(n), '#a78bfa');
      return;
    }
    if (n.kind === 'merge' && !n.pass) {
      const upto = n.out ? this.meta.S - 1 : n.fan - 1;
      box.append(UI.el('div', { class: 'kv' },
        UI.el('span', { class: 'k' }, '矩阵'), UI.el('span', { class: 'v' }, `${upto + 1}×${upto + 1} Sinkhorn(8)`),
        UI.el('span', { class: 'k' }, '读出行'), UI.el('span', { class: 'v' }, 'row ' + (n.out ? this.meta.S - 1 : n.li - 1)),
        UI.el('span', { class: 'k' }, '|g|'), UI.el('span', { class: 'v' }, UI.num(n.l2, 5)),
        UI.el('span', { class: 'k' }, '能量占比'), UI.el('span', { class: 'v' }, (n.share * 100).toFixed(2) + '%')));
      const list = UI.el('div', { class: 'exlist' });
      for (let j = 0; j <= upto; j++) {
        const le = this.layerE[j] || 0;
        list.append(UI.el('div', { class: 'exrow static' },
          UI.el('span', { class: 'i', style: { color: this.layerHue(j) } }, 'b' + j),
          UI.el('span', { class: 'b' }, UI.el('i', { style: { width: (this.total ? le / this.total * 100 : 0).toFixed(1) + '%', background: this.layerHue(j) } })),
          UI.el('span', { class: 'p' }, (this.total ? le / this.total * 100 : 0).toFixed(1) + '%')));
      }
      box.append(list, UI.el('div', { class: 'note' }, '读出行是 b₀…b' + upto + ' 上的可学习凸组合（双随机矩阵行和恒为 1）；条形为各源层当前梯度能量占比。'));
      this.spark(box, this.nodeSeries(n), '#fbbf24');
      return;
    }
    box.append(UI.el('div', { class: 'kv' },
      UI.el('span', { class: 'k' }, '合成 |g|'), UI.el('span', { class: 'v' }, UI.num(n.l2, 6)),
      UI.el('span', { class: 'k' }, '能量占比'), UI.el('span', { class: 'v' }, (n.share * 100).toFixed(2) + '%'),
      UI.el('span', { class: 'k' }, '权重数'), UI.el('span', { class: 'v' }, String(n.nw))));
    const wl = UI.el('div', { class: 'wlist' });
    for (const w of n.weights) {
      const v = Osc.l2(w);
      wl.append(UI.el('div', { class: 'wrow' },
        UI.el('span', { class: 'n', title: w }, Osc.alias(w)),
        UI.el('span', { class: 'v', style: v === 0 ? { color: 'var(--red)' } : {} }, UI.num(v, 5))));
    }
    box.append(wl);
    this.spark(box, this.nodeSeries(n), K.c);
  },
  coverage(li, e) {
    const down = `decoder_layers.${li}.moe.experts.${e}.down.weight`;
    const up = `decoder_layers.${li}.moe.experts.${e}.gate_up.weight`;
    let act = 0, streak = 0, best = 0;
    for (const vec of Osc.full) {
      const a = (vec[Osc._idx.get(down)] || 0) > 0 || (vec[Osc._idx.get(up)] || 0) > 0;
      if (a) { act++; streak = 0; } else { streak++; best = Math.max(best, streak); }
    }
    return { freq: Osc.full.length ? act / Osc.full.length : 0, streak: best };
  },
  renderStep() {
    const box = document.getElementById('arch-step');
    box.innerHTML = '';
    if (!Osc.steps.length) { box.append(UI.el('div', { class: 'empty' }, '无数据')); return; }
    const vec = Osc.vecAt(Osc.step);
    const zeros = vec.filter(v => v === 0).length;
    let act = 0, tot = 0, dead = [];
    for (const m of this.moe) { act += m.na; tot += m.n; }
    for (let li = 0; li < this.meta.S; li++) for (let e = 0; e < this.meta.E; e++) {
      const c = this.coverage(li, e);
      if (c.freq === 0) dead.push(`L${li}·e${e}`);
    }
    const top = this.bands.map(b => ({ li: b.li, e: this.layerE[b.li] || 0 })).sort((a, b) => b.e - a.e).slice(0, 3);
    box.append(
      UI.el('div', { class: 'kv' },
        UI.el('span', { class: 'k' }, 'step'), UI.el('span', { class: 'v' }, String(Osc.steps[Osc.step])),
        UI.el('span', { class: 'k' }, '合成 |g|'), UI.el('span', { class: 'v' }, UI.num(Math.sqrt(this.total), 5)),
        UI.el('span', { class: 'k' }, '零梯度权重'), UI.el('span', { class: 'v', style: zeros ? { color: 'var(--red)' } : {} }, String(zeros)),
        UI.el('span', { class: 'k' }, '激活专家'), UI.el('span', { class: 'v' }, `${act} / ${tot}`),
        UI.el('span', { class: 'k' }, '全窗口死亡专家'), UI.el('span', { class: 'v', style: dead.length ? { color: 'var(--red)' } : { color: 'var(--green)' } }, dead.length ? dead.length + ' 个' : '0')));
    if (dead.length) box.append(UI.el('div', { class: 'deadlist' }, dead.slice(0, 12).join(' ') + (dead.length > 12 ? ' …' : '')));
    const dep = UI.el('div', { class: 'dep' });
    const mx = Math.max(1e-9, ...this.layerE);
    this.layerE.forEach((e, li) => dep.append(UI.el('i', {
      title: `L${li} 能量 ${(e / (this.total || 1) * 100).toFixed(1)}%`,
      style: { height: Math.max(3, e / mx * 40) + 'px', background: this.layerHue(li) },
    })));
    box.append(UI.el('div', { class: 'dep-h' }, '逐层梯度能量（输入 → 输出）'), dep);
    box.append(UI.el('div', { class: 'kv', style: { marginTop: '8px' } },
      top.map(t => [UI.el('span', { class: 'k' }, '能量 top · L' + t.li), UI.el('span', { class: 'v' }, (t.e / (this.total || 1) * 100).toFixed(1) + '%')]).flat()));
  },
  renderLegend(box) {
    const names = { l2: '梯度 L2', share: '步内能量占比', load: 'MoE 负载代理', active: '激活 / 零梯度率', zero: '零梯度占比' };
    box.append(
      UI.el('span', {}, names[this.metric]),
      UI.el('span', { class: 'bar', style: { background: Osc.rampCSS() } }),
      UI.el('span', { class: 'lg-zero' }, UI.el('i', {}), '未路由 / 零梯度'),
      UI.el('span', { class: 'lg-hungry' }, UI.el('i', {}), `饥饿 <${this.hungry}%`));
  },

  // ---- interaction ------------------------------------------------------
  hit(ev) {
    const r = document.getElementById('arch-canvas').getBoundingClientRect();
    const x = (ev.clientX - r.left - this.cam.x) / this.cam.z;
    const y = (ev.clientY - r.top - this.cam.y) / this.cam.z;
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i];
      if (x >= n.x && x <= n.x + n.w && y >= n.y && y <= n.y + n.h) return n;
    }
    return null;
  },
  bind() {
    const $ = id => document.getElementById(id);
    const wrap = $('arch-wrap'), canvas = $('arch-canvas');
    const tip = this._tip = UI.el('div', { class: 'ctip' });
    document.body.append(tip);
    const on = (id, ev, fn) => $(id).addEventListener(ev, fn);
    on('arch-metric', 'change', e => { this.metric = e.target.value; this._dirty = true; Osc.renderLegend(); });
    on('arch-flow', 'change', e => { this.flow = e.target.value; });
    on('arch-speed', 'input', e => { this.speed = +e.target.value / 100; $('arch-speed-label').textContent = this.speed.toFixed(1) + '×'; });
    on('arch-cell', 'change', e => { this.cell = e.target.value; this._dirty = true; });
    on('arch-pow', 'input', e => { this.pow = +e.target.value / 100; $('arch-pow-label').textContent = 'g^' + this.pow.toFixed(2); this.stats(); this._dirty = true; this.renderInspector(); Moe.invalidate(); });
    on('arch-hungry', 'input', e => { this.hungry = +e.target.value; $('arch-hungry-label').textContent = this.hungry + '%'; this.stats(); this._dirty = true; Osc.renderLegend(); Moe.invalidate(); });
    on('arch-experts', 'change', e => { this.experts = e.target.checked; this.build(); this._dirty = true; });
    on('arch-labels', 'change', e => { this.labels = e.target.checked; this._dirty = true; });
    on('arch-bus', 'change', e => { this.bus = e.target.checked; this._dirty = true; });
    on('arch-focus', 'change', e => {
      this.focus = e.target.value; this._dirty = true;
      if (this.focus !== 'all') {
        const b = this.bands[+this.focus];
        this.cam.tz = Math.max(this.cam.tz, 0.85);
        this.cam.ty = -b.y0 * this.cam.tz + 30; this.syncZoomUI();
      }
    });
    on('arch-zoom', 'input', e => {
      const z = +e.target.value / 100;
      const cx = this.vw / 2, cy = this.vh / 2;
      const wx = (cx - this.cam.tx) / this.cam.tz, wy = (cy - this.cam.ty) / this.cam.tz;
      this.cam.tz = z; this.cam.tx = cx - wx * z; this.cam.ty = cy - wy * z;
      $('arch-zoom-label').textContent = Math.round(z * 100) + '%';
    });
    on('arch-fit', 'click', () => this.fitWidth());
    on('arch-fitall', 'click', () => this.fitAll());
    on('arch-reset', 'click', () => {
      this.metric = 'l2'; this.flow = 'both'; this.speed = 1; this.cell = 'load'; this.pow = 2; this.hungry = 4;
      this.experts = true; this.labels = true; this.bus = true; this.focus = 'all';
      $('arch-metric').value = 'l2'; $('arch-flow').value = 'both'; $('arch-speed').value = '100'; $('arch-speed-label').textContent = '1.0×';
      $('arch-cell').value = 'load'; $('arch-pow').value = '200'; $('arch-pow-label').textContent = 'g^2.00';
      $('arch-hungry').value = '4'; $('arch-hungry-label').textContent = '4%';
      $('arch-experts').checked = true; $('arch-labels').checked = true; $('arch-bus').checked = true; $('arch-focus').value = 'all';
      this.build(); this.fitWidth(); Osc.renderLegend(); this.renderInspector();
    });
    on('arch-png', 'click', () => this.exportPNG());
    on('arch-json', 'click', () => this.exportJSON());
    canvas.addEventListener('wheel', ev => {
      ev.preventDefault();
      const r = canvas.getBoundingClientRect();
      const mx = ev.clientX - r.left, my = ev.clientY - r.top;
      const wx = (mx - this.cam.tx) / this.cam.tz, wy = (my - this.cam.ty) / this.cam.tz;
      const z = Math.min(2.4, Math.max(0.12, this.cam.tz * Math.exp(-ev.deltaY * 0.0012)));
      this.cam.tz = z; this.cam.tx = mx - wx * z; this.cam.ty = my - wy * z;
      this.syncZoomUI();
    }, { passive: false });
    canvas.addEventListener('pointerdown', ev => {
      try { canvas.setPointerCapture(ev.pointerId); } catch { /* synthetic or released pointer */ }
      this._drag = { x: ev.clientX, y: ev.clientY, cx: this.cam.tx, cy: this.cam.ty, moved: false };
    });
    canvas.addEventListener('pointermove', ev => {
      if (this._drag) {
        const dx = ev.clientX - this._drag.x, dy = ev.clientY - this._drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) this._drag.moved = true;
        if (this._drag.moved) {
          this.cam.tx = this._drag.cx + dx; this.cam.ty = this._drag.cy + dy;
          this.cam.x = this.cam.tx; this.cam.y = this.cam.ty;
          // drag bypasses the damping loop, so the cached static scene must be
          // invalidated here or nodes freeze while particles keep moving
          this._dirty = true;
          tip.style.display = 'none';
          return;
        }
      }
      const n = this.hit(ev);
      if ((n && n.id) !== (this.hover && this.hover.id)) { this.hover = n; this._dirty = true; }
      canvas.style.cursor = n ? 'pointer' : 'grab';
      if (!n) { tip.style.display = 'none'; return; }
      const m = n.kind === 'expert' || n.kind === 'router' ? this.moe[n.li] : null;
      tip.innerHTML = `<b>${UI.esc(n.kind === 'expert' ? `L${n.li} expert ${n.e}` : n.title)}</b><span class="t2">${UI.esc(this.KIND[n.kind].t)} · |g| ${UI.num(n.l2, 5)} · 能量 ${(n.share * 100).toFixed(2)}%${n.kind === 'expert' ? ` · ${n.active ? '负载 ' + (n.load * 100).toFixed(1) + '%' : '未路由'}` : ''}${n.kind === 'router' && m ? ` · 激活 ${m.na}/${m.n} · eff ${m.eff.toFixed(1)}` : ''}</span>`;
      tip.style.display = 'block';
      tip.style.left = Math.min(window.innerWidth - 320, ev.clientX + 14) + 'px';
      tip.style.top = (ev.clientY + 14) + 'px';
    });
    canvas.addEventListener('pointerup', ev => {
      const drag = this._drag; this._drag = null;
      if (drag && !drag.moved) {
        const n = this.hit(ev);
        if (n) { this.sel = n; this._dirty = true; this.renderInspector(); }
      }
    });
    canvas.addEventListener('pointerleave', () => { tip.style.display = 'none'; if (this.hover) { this.hover = null; this._dirty = true; } });
    const mini = $('arch-mini');
    const jump = ev => {
      const r = mini.getBoundingClientRect();
      const sy = (this.vh - 8) / this.world.h, sx = (56 - 8) / this.world.w;
      const wy = (ev.clientY - r.top - 4) / sy, wx = (ev.clientX - r.left - 4) / sx;
      this.cam.tx = this.vw / 2 - wx * this.cam.tz; this.cam.ty = this.vh / 2 - wy * this.cam.tz;
    };
    mini.addEventListener('pointerdown', ev => { try { mini.setPointerCapture(ev.pointerId); } catch { } this._mini = true; jump(ev); });
    mini.addEventListener('pointermove', ev => { if (this._mini) jump(ev); });
    mini.addEventListener('pointerup', () => { this._mini = false; });
    window.addEventListener('resize', UI.debounce(() => { if (this.visible) { this.resize(); this._dirty = true; } }, 150));
    Osc.sub(kind => {
      if (kind === 'view') {
        this.visible = !document.getElementById('grad-sub-arch').hidden;
        if (this.visible) { this.render(); } else this.stop();
      } else if (kind === 'data') { this.setMeta(App.job); this._fitted = false; this.render(); }
      else if (kind === 'rebuild') { if (this.visible) this.render(); }
      else if (kind === 'step') { if (this.visible) { this.stats(); this._dirty = true; this.renderInspector(); this.renderStep(); } }
    });
  },
  exportPNG() {
    const s = 1.5;
    const c = document.createElement('canvas');
    c.width = Math.round(this.world.w * s); c.height = Math.round(this.world.h * s);
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0a0e14'; ctx.fillRect(0, 0, c.width, c.height);
    this.stats();
    this.scene(ctx, s, { x: 0, y: 0, z: 1 });
    const a = document.createElement('a');
    a.href = c.toDataURL('image/png');
    a.download = `arch-flow-${App.currentJob}-step${Osc.steps[Osc.step] ?? 0}.png`;
    a.click();
    UI.toast('架构数据流图已导出 PNG');
  },
  exportJSON() {
    this.stats();
    const data = {
      job: App.currentJob, step: Osc.steps[Osc.step], rank: Osc.rank, meta: this.meta,
      total_energy: this.total,
      nodes: this.nodes.filter(n => n.kind !== 'expert').map(n => ({ id: n.id, kind: n.kind, li: n.li, l2: n.l2, energy: n.energy, share: n.share, zero: n.zero })),
      moe: this.moe.map(m => ({
        layer: m.li, active: m.na, experts: m.n, entropy_norm: m.nH, effective_experts: m.eff, cv: m.cv,
        load_proxy: m.shares, routed: m.act,
      })),
      note: 'load_proxy = normalize(grad_l2^p) over experts; zero grad means no tokens routed this step.',
    };
    UI.download(`arch-flow-${App.currentJob}-step${Osc.steps[Osc.step] ?? 0}.json`, JSON.stringify(data, null, 2), 'application/json');
    UI.toast('节点指标已导出 JSON');
  },
};