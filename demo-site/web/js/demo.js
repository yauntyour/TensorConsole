// Canvas chrome palette resolved from the CSS tokens, so an appearance change
// reaches the 2D contexts too. Data-encoding colors (module hues, flux ramp,
// zero-grad red, hungry amber) stay fixed — they carry meaning, not theme.
const TC = {
  bg: '#0a0d12', canvasBg: '#0b0f14', panel: '#121820', panel3: '#1c2530',
  line: '#26313e', groupHead: '#0e151d', nullCell: '#10161d',
  text: '#e8eef4', muted: '#93a1b0', dim: '#64748b',
};

function resolvePalette() {
  const probe = document.createElement('div');
  probe.style.cssText = 'position:absolute;visibility:hidden;pointer-events:none';
  document.body.append(probe);
  // custom properties hold unresolved color-mix() text; read them back as
  // computed colors so canvas fillStyle always gets a parseable value
  const read = (expr, fallback) => {
    probe.style.backgroundColor = '';
    probe.style.backgroundColor = expr;
    return getComputedStyle(probe).backgroundColor || fallback;
  };
  TC.bg = read('var(--bg)', TC.bg);
  TC.canvasBg = read('var(--canvas-bg)', TC.canvasBg);
  TC.panel = read('var(--panel)', TC.panel);
  TC.panel3 = read('var(--panel-3)', TC.panel3);
  TC.line = read('var(--line-strong)', TC.line);
  TC.groupHead = read('color-mix(in srgb, var(--canvas-bg) 82%, var(--site-surface))', TC.groupHead);
  TC.nullCell = read('color-mix(in srgb, var(--canvas-bg) 88%, var(--site-surface))', TC.nullCell);
  TC.text = read('var(--text)', TC.text);
  TC.muted = read('var(--muted)', TC.muted);
  TC.dim = read('var(--dim)', TC.dim);
  probe.remove();
}

// Offline replay layer for the public demo.
// Serves one recorded tcd job through the same API surface the SPA already uses,
// so every visualization module runs unchanged without a daemon.
const Demo = {
  meta: null,
  cache: new Map(),
  _stepped: false,

  async data(name) {
    if (this.cache.has(name)) return this.cache.get(name);
    const r = await fetch('/data/' + name);
    if (!r.ok) throw new Error('回放数据缺失：' + name);
    const j = await r.json();
    this.cache.set(name, j);
    return j;
  },

  async resolve(method, path) {
    const u = new URL(path, 'http://demo.local');
    const p = u.pathname.replace(/\/+$/, '');
    const q = u.searchParams;
    if (method !== 'GET') throw new Error('离线回放只读：该操作需要本地 tcd 守护进程');

    if (p === '/api/system') return { hostname: '离线回放', os: 'recorded', cpu_cores: 0, host: {} };
    if (p === '/api/gpus') return { gpus: [], host: {}, cluster: {}, history: [] };
    if (p === '/api/projects') return this.data('projects.json');
    if (p === '/api/jobs') return this.data('jobs.json');
    if (p === '/api/docker/ps' || p === '/api/docker/images') return { available: false, rows: [] };
    if (/^\/api\/projects\/[^/]+\/schema$/.test(p)) return { schema: { groups: [] } };

    const m = p.match(/^\/api\/jobs\/([^/]+)(\/.*)?$/);
    if (!m) throw new Error('离线回放未提供该接口：' + p);
    const sub = m[2] || '';

    if (sub === '') return this.data('job.json');
    if (sub === '/metrics') return this.data('metrics.json');
    if (sub === '/grad/order') return this.data('order.json');
    if (sub === '/artifacts') return [];
    if (sub === '/profile') return null;
    if (sub === '/logs') {
      const all = await this.data('logs.json');
      const off = +(q.get('offset') || 0);
      const lim = +(q.get('limit') || 2000);
      const lines = all.lines.slice(off, off + lim);
      return { lines, next_offset: off + lines.length, total: all.total };
    }
    if (sub === '/grad') {
      const rows = await this.data('grad.json');
      const rank = q.get('rank');
      const step = q.get('step');
      const from = q.get('from'), to = q.get('to');
      let out = rows;
      if (rank !== null && rank !== '') out = out.filter(r => String(r.rank ?? 0) === rank);
      if (step !== null && step !== '') out = out.filter(r => String(r.step) === step);
      if (from) out = out.filter(r => r.step >= +from);
      if (to) out = out.filter(r => r.step <= +to);
      return out;
    }
    if (sub === '/report') {
      const rep = await this.data('report.json');
      return q.get('format') === 'json' ? rep.json : rep.md;
    }
    throw new Error('离线回放未提供该接口：' + p);
  },

  install() {
    API.req = (method, path) => this.resolve(method, path);
    API.text = async path => {
      const v = await this.resolve('GET', path);
      return typeof v === 'string' ? v : JSON.stringify(v, null, 2);
    };
    API.events = () => ({ close() { } });
    API.artifactUrl = () => '#';
    this.ready = this.data('meta.json').then(m => { this.meta = m; return m; });
  },

  // ------------------------------------------------------------ UI adaptation
  adapt() {
    const $ = id => document.getElementById(id);
    const hide = el => { if (el) el.hidden = true; };

    // The demo is one recorded run: drop the fleet-management pages.
    hide($('nav'));
    $('job-nav').hidden = false;
    hide($('token-btn'));
    for (const page of ['overview', 'jobs', 'new', 'projects', 'docker']) hide($('page-' + page));
    for (const a of document.querySelectorAll('[data-tab="prof"]')) a.style.display = 'none';
    hide($('pane-prof'));
    const follow = $('logs-follow');
    if (follow) follow.closest('.switch').style.display = 'none';

    const strip = UI.el('div', { id: 'demo-strip' },
      UI.el('span', { class: 'demo-tag' }, '离线回放'),
      UI.el('span', { class: 'demo-txt' },
        '这是一次真实训练的完整录像：MetaModel（10 层 × 8 专家，301 个权重）跑了 900 步后被优雅停止。',
        UI.el('b', {}, '拖动下方 step 滑杆或点「▶ 回放」'), '，就能逐 step 重看模型内部的梯度流动。'),
      UI.el('span', { class: 'spacer' }),
      UI.el('button', { class: 'btn sm', id: 'demo-help' }, '这是什么？'));
    $('topbar').after(strip);
    $('demo-help').addEventListener('click', () => this.guide());

    // Route everything to the recorded run.
    const origRoute = App.route.bind(App);
    App.route = () => {
      const parts = decodeURIComponent(location.hash.replace(/^#/, '')).split('/').filter(Boolean);
      if (parts[0] === 'job' && parts[1]) {
        if (parts[2] === 'prof') { location.hash = `#job/${parts[1]}/grad/arch`; return; }
        origRoute();
        return;
      }
      const go = m => { location.hash = `#job/${m.job_id}/grad/arch`; };
      if (this.meta) go(this.meta); else this.ready.then(go);
    };

    // Land on the final recorded step, where routing collapse is visible.
    const origGrad = App.reloadGrad.bind(App);
    App.reloadGrad = async () => {
      await origGrad();
      if (!this._stepped && Osc.steps.length) {
        this._stepped = true;
        const m = this.meta || await this.ready;
        Osc.setStep(Math.min(m.default_step, Osc.steps.length - 1));
      }
    };

    const origTab = App.switchTab.bind(App);
    App.switchTab = (tab, force) => origTab(tab === 'prof' ? 'grad' : tab, force);

    // Read-only run: strip the mutation controls from the job header.
    const origHeader = App.renderJobHeader.bind(App);
    App.renderJobHeader = () => {
      origHeader();
      const box = $('job-header');
      box.querySelectorAll('button').forEach(b => { if (b.textContent.trim() !== '刷新') b.remove(); });
      box.querySelectorAll('.edit').forEach(e => e.remove());
    };

    const origInit = App.init.bind(App);
    App.init = async () => { await origInit(); this.afterInit(); };
  },

  afterInit() {
    const conn = document.getElementById('conn');
    conn.className = 'pill demo';
    document.getElementById('conn-text').textContent = '录制数据 · 只读';
    document.getElementById('crumb').innerHTML = '训练回放';
    if (!localStorage.getItem('tc_demo_seen')) this.guide();
  },

  guide() {
    localStorage.setItem('tc_demo_seen', '1');
    const row = (k, v) => UI.el('div', { class: 'guide-row' },
      UI.el('span', { class: 'guide-k' }, k), UI.el('span', { class: 'guide-v' }, v));
    UI.modal({
      title: '把一次 AI 模型训练拆开看',
      wide: true,
      body: [
        UI.el('p', { class: 'guide-lead' },
          '训练大模型时，最难回答的问题是「模型内部到底在学什么、哪里学坏了」。',
          'TensorConsole 把训练过程录下来，让你像看录像一样逐 step 回放。',
          '这里播放的是一次真实运行：10 层 × 8 专家的 MetaModel，900 步，301 个权重全部采集。'),
        UI.el('div', { class: 'guide-grid' },
          row('架构数据流', '模型的真实前向结构画成节点图。青绿粒子是前向数据，琥珀粒子是反向梯度。' +
            '每个 MoE 节点内嵌 8 个专家格，斜线 = 这一步没有 token 路由到它，琥珀边 = 长期饥饿。'),
          row('MoE 路由', '8 个专家谁在干活、谁在摸鱼。层 × 专家矩阵 + 均衡度趋势 + 专家覆盖表，' +
            '能直接看出路由坍缩（某一层只剩一个专家在干活）。'),
          row('权重热力图', '301 个权重 × 900 步的梯度强度。红色刻痕是零梯度，' +
            '也就是这一步完全没学到东西的权重。')),
        UI.el('p', { class: 'guide-note' },
          '上手只要一步：拖动 step 滑杆，或点「▶ 回放」让它自己播。三个视图共享同一个 step，切换视图不会丢进度。'),
        UI.el('div', { class: 'guide-find' },
          UI.el('b', {}, '这次录像里实际能看到的问题'),
          UI.el('span', {},
            'step 1–60：80 个专家全部在接 token，路由均衡。',
            'step 300：53 个专家在这一步完全没被路由到。',
            'step 900：仍有 36 个专家零梯度（301 个权重里 72 个为零），',
            'L3 / L4 / L5 的梯度能量几乎 100% 集中在单个专家上。'),
          UI.el('em', {}, '这就是 MoE 路由坍缩 —— 名义上 8 个专家，实际只有 1 个在学。' +
            '零梯度判定是精确的；能量占比为梯度代理指标，真实 token 计数需引擎侧 dispatch 遥测。')),
      ],
      actions: [{ label: '开始回放', kind: 'primary', onClick: c => { c(); Osc.play(true); } },
      { label: '我自己看', kind: 'ghost', onClick: c => c() }],
    });
  },
};

Demo.install();
resolvePalette();
document.addEventListener('DOMContentLoaded', () => { resolvePalette(); Demo.adapt(); });

// Nudge writes root variables; repaint the canvases with the new chrome colors.
new MutationObserver(() => {
  resolvePalette();
  if (typeof Osc !== 'undefined' && Osc.steps.length) Osc.renderAll();
}).observe(document.documentElement, { attributes: true, attributeFilter: ['style', 'class'] });