// TensorConsole SPA controller
const App = {
  page: 'overview', tab: 'metrics', currentJob: null, job: null,
  projects: [], jobs: [], metricsRows: [], gradOrder: [],
  ov: { samples: [], charts: [], gpuKey: '', hostName: '', cpuCores: 0 },
  newType: 'train', newValues: {}, newForm: null, deployForm: null,
  logsOffset: 0, logsTotal: 0, _dockerTimer: null, _sse: null,

  // ------------------------------------------------------------ boot
  async init() {
    document.querySelectorAll('#nav a, #job-nav a').forEach(a =>
      a.addEventListener('click', () => this.showPage(a.dataset.page, a.dataset.tab)));
    document.getElementById('token-btn').addEventListener('click', async () => {
      const v = await UI.prompt('访问令牌', [{ key: 'token', label: 'Bearer Token', value: API.token, placeholder: '服务端未配置可留空', help: '非 loopback 监听时服务端要求令牌' }], '应用');
      if (v) { API.setToken(v.token); UI.toast('令牌已保存'); location.reload(); }
    });
    document.getElementById('ov-jobs-all').addEventListener('click', () => this.showPage('jobs'));
    for (const id of ['f-status', 'f-type', 'f-project'])
      document.getElementById(id).addEventListener('change', () => this.loadJobs());
    document.getElementById('f-search').addEventListener('input', UI.debounce(() => this.renderJobs(), 200));
    document.getElementById('jobs-refresh').addEventListener('click', () => this.loadJobs());
    document.getElementById('jobs-new').addEventListener('click', () => this.showPage('new'));
    document.getElementById('proj-refresh').addEventListener('click', () => this.loadProjects());
    document.getElementById('proj-new').addEventListener('click', () => this.projectModal());
    document.getElementById('new-project').addEventListener('change', () => this.loadSchema());
    document.getElementById('new-submit').addEventListener('click', () => this.submitNew());
    document.getElementById('docker-refresh').addEventListener('click', () => this.loadDocker());
    document.getElementById('docker-auto').addEventListener('change', e => {
      clearInterval(this._dockerTimer);
      if (e.target.checked) this._dockerTimer = setInterval(() => { if (this.page === 'docker') this.loadDocker(); }, 10000);
    });
    document.getElementById('deploy-submit').addEventListener('click', () => this.submitDeploy());
    document.querySelectorAll('#job-tabs a').forEach(a =>
      a.addEventListener('click', () => this.switchTab(a.dataset.tab)));
    document.getElementById('logs-follow').addEventListener('change', () => { });
    document.getElementById('logs-reload').addEventListener('click', () => this.loadLogs(true));
    document.getElementById('logs-bottom').addEventListener('click', () => {
      const v = document.getElementById('logs-view'); v.scrollTop = v.scrollHeight;
    });
    document.getElementById('report-reload').addEventListener('click', () => { this.loadReport(); this.loadArtifacts(); });
    document.getElementById('report-md-dl').addEventListener('click', async () => {
      UI.download(this.currentJob + '.md', await API.text('/api/jobs/' + this.currentJob + '/report?format=md'), 'text/markdown');
    });
    document.getElementById('report-json-dl').addEventListener('click', async () => {
      UI.download(this.currentJob + '.json', await API.text('/api/jobs/' + this.currentJob + '/report?format=json'), 'application/json');
    });
    Osc.bind();
    Arch.bind();
    Moe.bind();
    Flame.bind();
    this.connectSSE();
    await this.loadProjects();
    window.addEventListener('hashchange', () => this.route());
    this.route();
  },

  // deep links: #overview | #jobs | #job/<id>/<tab>[/<grad-subview>]
  route() {
    const h = decodeURIComponent(location.hash.replace(/^#/, ''));
    const parts = h.split('/').filter(Boolean);
    if (parts[0] === 'job' && parts[1]) {
      const tab = parts[2] || 'metrics';
      const sub = parts[3];
      const go = () => {
        this.showPage('job', tab);
        if (tab === 'grad' && sub && ['arch', 'moe', 'heat'].includes(sub)) Osc.setSubview(sub);
      };
      if (this.currentJob === parts[1]) go();
      else this.openJob(parts[1]).then(go);
      return;
    }
    this.showPage(['overview', 'jobs', 'new', 'projects', 'docker'].includes(parts[0]) ? parts[0] : 'overview');
  },
  setHash(h) {
    if (location.hash !== '#' + h) history.replaceState(null, '', '#' + h);
  },

  connectSSE() {
    const conn = document.getElementById('conn');
    const setConn = ok => {
      conn.className = 'pill ' + (ok ? 'on' : 'off');
      document.getElementById('conn-text').textContent = ok ? 'SSE 已连接' : 'SSE 断开';
    };
    this._sse = API.events('', {
      hello: () => setConn(true),
      heartbeat: () => setConn(true),
      gpu: d => this.onGpu(d),
      'job.status': d => {
        this.feed('job', `${d.job_id} → ${d.status}`);
        if (this.page === 'jobs') this.loadJobs();
        if (d.job_id === this.currentJob) {
          this.refreshJobHeader();
          if (['succeeded', 'failed', 'stopped'].includes(d.status)) this.switchTab(this.tab, true);
        }
      },
      'job.log': d => {
        if (d.job_id === this.currentJob && this.tab === 'logs' &&
          document.getElementById('logs-follow').checked) this.appendLog(d.line);
      },
      metrics: d => {
        if (d.job_id !== this.currentJob) return;
        this.feed('metrics', `step ${d.step ?? '?'} r${d.rank ?? 0}`);
        this.pushMetric(d);
      },
      'grad.step': d => {
        if (d.job_id !== this.currentJob) return;
        this.feed('grad', `step ${d.step ?? '?'} r${d.rank ?? 0}`);
        if (this.tab === 'grad') this._gradReload();
      },
      'prof.ready': d => {
        if (d.job_id !== this.currentJob) return;
        this.feed('job', 'profile ready');
        if (this.tab === 'prof') this.reloadProfile();
      },
    }, setConn);
    this._gradReload = UI.throttle(() => this.reloadGrad(), 1500);
    this._metricRender = UI.throttle(() => this.renderMetrics(), 1200);
    // read the newest sample at call time — never close over an event payload
    this._ovCards = UI.throttle(() => { if (this.page === 'overview') this.renderOvLive(); }, 1500);
    this._ovJobs = UI.throttle(() => {
      if (this.page !== 'overview') return;
      API.get('/api/jobs?limit=6').then(j => this.renderOvJobs(j)).catch(() => { });
    }, 5000);
  },

  feed(kind, msg) {
    const box = document.getElementById('ev-feed');
    const row = UI.el('div', { class: 'ev' },
      UI.el('span', { class: 't' }, UI.clock(Date.now())),
      UI.el('span', { class: 'k ' + kind }, kind),
      UI.el('span', { class: 'm' }, msg));
    box.prepend(row);
    while (box.children.length > 40) box.lastChild.remove();
  },

  // ------------------------------------------------------------ routing
  showPage(name, tab) {
    this.page = name;
    document.querySelectorAll('.page').forEach(p => { p.hidden = p.id !== 'page-' + name; });
    document.querySelectorAll('#nav a').forEach(a => a.classList.toggle('active', a.dataset.page === name));
    const titles = { overview: '总览', jobs: '任务', new: '新建任务', projects: '项目', docker: 'Docker / 部署', job: '任务详情' };
    document.getElementById('crumb').innerHTML = UI.esc(titles[name] || name) +
      (name === 'job' && this.job ? ` <span>/ ${UI.esc(this.job.title || this.job.id)}</span>` : '');
    if (name === 'overview') this.loadOverview();
    else if (name === 'jobs') this.loadJobs();
    else if (name === 'projects') this.loadProjects();
    else if (name === 'new') this.loadNew();
    else if (name === 'docker') this.loadDocker();
    else if (name === 'job') {
      if (!this.currentJob) { this.showPage('jobs'); return; }
      this.switchTab(tab || this.tab);
    }
    if (name !== 'job') this.setHash(name);
    document.getElementById('content').scrollTop = 0;
  },

  // ------------------------------------------------------------ overview
  async loadOverview() {
    try {
      const [sys, g, jobs] = await Promise.all([
        API.get('/api/system'), API.get('/api/gpus'), API.get('/api/jobs?limit=6'),
      ]);
      this.ov.hostName = sys.hostname || '';
      this.ov.cpuCores = sys.cpu_cores || 0;
      this.ov.samples = (g.history || []).slice(-300);
      if (!this.ov.samples.length)
        this.ov.samples = [{ gpus: g.gpus || [], host: g.host || sys.host || {}, cluster: g.cluster || {} }];
      this.renderOvLive();
      this.buildOvCharts();
      this.renderOvJobs(jobs);
    } catch (e) { UI.err(e); }
  },
  ovSample() {
    return this.ov.samples[this.ov.samples.length - 1] || { gpus: [], host: {}, cluster: {} };
  },
  renderOvLive() {
    const s = this.ovSample();
    this.renderHost(s.host || {});
    this.renderCluster(s.cluster || {}, s.gpus || []);
    this.renderGpus(s.gpus || []);
  },
  // nvidia-smi reports unavailable metrics as -1; render those as a dash
  mnum(v, d = 0) { return UI.num(typeof v === 'number' && v >= 0 ? v : undefined, d); },
  statTile(k, v, unit, meter, sub) {
    const pct = typeof meter === 'number' && meter >= 0 ? Math.min(100, meter) : null;
    return UI.el('div', { class: 'stat' },
      UI.el('span', { class: 'k' }, k),
      UI.el('span', { class: 'v' }, v, unit ? UI.el('small', {}, unit) : null),
      sub ? UI.el('span', { class: 'sub' }, sub) : null,
      pct === null ? null : UI.el('div', { class: 'meter' }, UI.el('i', {
        class: pct > 85 ? 'hot' : pct > 65 ? 'warn' : '', style: { width: pct + '%' },
      })));
  },
  renderHost(h) {
    const gb = v => (typeof v === 'number' && v >= 0 ? UI.num(v / 1024, 1) : '—');
    document.getElementById('ov-host-sub').textContent = this.ov.hostName || '';
    const box = document.getElementById('ov-host');
    box.innerHTML = '';
    box.append(
      this.statTile('CPU 使用率', this.mnum(h.cpu_pct, 1), '%', h.cpu_pct,
        this.ov.cpuCores ? `${this.ov.cpuCores} 逻辑核心` : ''),
      this.statTile('内存使用率', this.mnum(h.mem_pct, 1), '%', h.mem_pct,
        `${gb(h.mem_used_mb)} / ${gb(h.mem_total_mb)} GB`));
  },
  renderCluster(c, gpus) {
    const n = c.count ?? gpus.length;
    document.getElementById('ov-cluster-sub').textContent = n ? `${n} 卡 · ${gpus[0]?.name || ''}` : '无设备';
    document.getElementById('ov-gpus-sub').textContent = n ? `${n} 张显卡` : '无设备';
    const box = document.getElementById('ov-cluster');
    box.innerHTML = '';
    box.append(
      this.statTile('平均利用率', this.mnum(c.util, 1), '%', c.util,
        c.util_max >= 0 ? `单卡峰值 ${this.mnum(c.util_max, 0)}%` : ''),
      this.statTile('显存合计', this.mnum(c.mem_used, 0), 'MiB', c.mem_pct,
        c.mem_total > 0 ? `共 ${this.mnum(c.mem_total, 0)} MiB · ${this.mnum(c.mem_pct, 1)}%` : ''),
      this.statTile('总功耗', this.mnum(c.power, 0), 'W', null),
      this.statTile('峰值温度', this.mnum(c.temp_max, 0), '°C', null));
  },
  renderGpus(gpus) {
    this._lastGpus = gpus;
    this._procAll = this._procAll || {};
    const box = document.getElementById('ov-gpus');
    box.innerHTML = '';
    if (!gpus.length) {
      box.append(UI.el('div', { class: 'card' }, UI.el('div', { class: 'empty' }, UI.el('b', {}, '未检测到 GPU'), 'nvidia-smi 不可用或无设备')));
      return;
    }
    for (const g of gpus) {
      const procs = [...(g.processes || [])].sort((a, b) => (b.job ? 1 : 0) - (a.job ? 1 : 0) || (b.used_memory || 0) - (a.used_memory || 0));
      const all = !!this._procAll[g.index];
      const shown = all ? procs : procs.slice(0, 4);
      const procRow = p => UI.el('tr', {},
        UI.el('td', { class: 'mono dim' }, 'pid ' + p.pid),
        UI.el('td', { class: 'num' }, p.used_memory >= 0 ? UI.num(p.used_memory, 0) + ' MiB' : '—'),
        UI.el('td', {}, p.job ? UI.el('span', { class: 'j', onclick: () => this.openJob(p.job.job_id || p.job.id) }, p.job.title || p.job.job_id || p.job.id) : UI.el('span', { class: 'dim' }, '外部进程')));
      const procTbl = UI.el('table', { class: 'proc-tbl' }, shown.map(procRow));
      const more = procs.length > 4 ? UI.el('button', {
        class: 'btn sm ghost', style: { marginTop: '4px' },
        onclick: () => { this._procAll[g.index] = !all; this.renderGpus(this._lastGpus); },
      }, all ? '收起' : `展开其余 ${procs.length - 4} 个进程`) : null;
      const memPct = g.mem_total > 0 && g.mem_used >= 0 ? (g.mem_used / g.mem_total) * 100 : -1;
      box.append(UI.el('div', { class: 'card' },
        UI.el('div', { class: 'card-h' }, `GPU ${g.index}`, UI.el('span', { class: 'sub' }, g.name),
          UI.el('span', { class: 'right' },
            UI.el('span', { class: 'tag' }, this.mnum(g.temp, 0) + '°C'),
            UI.el('span', { class: 'tag' }, this.mnum(g.power, 0) + 'W'))),
        UI.el('div', { class: 'card-b stats' },
          this.statTile('利用率', this.mnum(g.util, 0), '%', g.util),
          this.statTile('显存', this.mnum(g.mem_used, 0), `/ ${this.mnum(g.mem_total, 0)} MiB`, memPct),
          procs.length ? UI.el('div', { style: { gridColumn: '1 / -1' } },
            UI.el('div', { style: { color: 'var(--dim)', fontSize: '11px', marginBottom: '2px' } }, `计算进程 ${procs.length}`),
            procTbl, more) : null)));
    }
  },
  buildOvCharts() {
    Charts.destroyIn(document.getElementById('page-overview'));
    this.ov.charts = [];
    const idxs = (this.ovSample().gpus || []).map(g => g.index);
    this.ov.gpuKey = idxs.join(',');
    const mk = (boxId, defs, unit) => {
      const h = Charts.line(document.getElementById(boxId), defs, { height: 170, yUnit: unit, xTime: true });
      this.ov.charts.push({ h, defs });
    };
    mk('ch-host', [
      { label: 'CPU', color: '#2dd4bf', pick: s => s.host?.cpu_pct },
      { label: '内存', color: '#fbbf24', pick: s => s.host?.mem_pct },
    ], '%');
    mk('ch-cluster', [
      { label: '平均利用率', color: '#60a5fa', pick: s => s.cluster?.util },
      { label: '显存占用', color: '#a78bfa', pick: s => s.cluster?.mem_pct },
    ], '%');
    mk('ch-util', idxs.map((i, k) => ({ label: 'GPU' + i, color: Charts.COLORS[k % 8], pick: s => (s.gpus || []).find(g => g.index === i)?.util })), '%');
    mk('ch-mem', idxs.map((i, k) => ({ label: 'GPU' + i, color: Charts.COLORS[k % 8], pick: s => (s.gpus || []).find(g => g.index === i)?.mem_used })), 'MiB');
    this.fillOvCharts();
  },
  fillOvCharts() {
    const xs = this.ov.samples.map((s, i) => (s.host?.ts || Date.now() - (this.ov.samples.length - i) * 2000) / 1000);
    for (const { h, defs } of this.ov.charts)
      h.setData(xs, defs.map(d => this.ov.samples.map(s => { const v = d.pick(s); return v === undefined || v === null || v < 0 ? null : v; })));
  },
  onGpu(d) {
    if (this.page !== 'overview') return;
    this.ov.samples.push({ gpus: d.gpus || [], host: d.host || {}, cluster: d.cluster || {} });
    if (this.ov.samples.length > 300) this.ov.samples.shift();
    const key = (d.gpus || []).map(g => g.index).join(',');
    if (key !== this.ov.gpuKey) this.buildOvCharts();
    else this.fillOvCharts();
    this._ovCards();
    this._ovJobs();
  },
  renderOvJobs(jobs) {
    const running = (jobs || []).filter(j => j.status === 'running').length;
    document.getElementById('ov-jobs-sub').textContent = `运行中 ${running} · 最近 ${jobs.length}`;
    const box = document.getElementById('ov-jobs');
    box.innerHTML = '';
    box.append(UI.table([
      { label: '任务', get: j => UI.el('span', { class: 'rowlink', onclick: () => this.openJob(j.id) }, j.title || j.id) },
      { label: '类型', get: j => UI.el('span', { class: 'tag' }, j.type) },
      { label: '状态', get: j => UI.badge(j.status) },
      { label: '创建', get: j => UI.el('span', { class: 'dim' }, UI.rel(j.created)) },
    ], jobs, '暂无任务 — 点击「新建任务」开始'));
  },

  // ------------------------------------------------------------ jobs
  async loadJobs() {
    try {
      const q = new URLSearchParams({ limit: '200' });
      for (const [id, key] of [['f-status', 'status'], ['f-type', 'type'], ['f-project', 'project']]) {
        const v = document.getElementById(id).value;
        if (v) q.set(key, v);
      }
      this.jobs = await API.get('/api/jobs?' + q);
      this.renderJobs();
    } catch (e) { UI.err(e); }
  },
  renderJobs() {
    const q = document.getElementById('f-search').value.trim().toLowerCase();
    const rows = this.jobs.filter(j => !q || (j.title || '').toLowerCase().includes(q) || j.id.toLowerCase().includes(q));
    document.getElementById('jobs-count').textContent = `${rows.length} / ${this.jobs.length} 个任务`;
    const box = document.getElementById('jobs-table');
    box.innerHTML = '';
    box.append(UI.table([
      {
        label: '任务', get: j => UI.el('div', {},
          UI.el('span', { class: 'rowlink', onclick: () => this.openJob(j.id) }, j.title || j.id),
          UI.el('div', { class: 'mono dim', style: { fontSize: '10.5px' } }, j.id))
      },
      { label: '项目', get: j => UI.el('span', { class: 'muted' }, j.project_id) },
      { label: '类型', get: j => UI.el('span', { class: 'tag' }, j.type) },
      { label: '状态', get: j => UI.badge(j.status) },
      { label: 'ranks', num: true, get: j => String((j.ranks || []).length || 1) },
      { label: '创建', get: j => UI.el('span', { class: 'dim' }, UI.rel(j.created)) },
      { label: '耗时', num: true, get: j => UI.dur(j.started, j.ended) },
      { label: '退出码', num: true, get: j => j.exit_code === null || j.exit_code === undefined ? '—' : String(j.exit_code) },
      {
        label: '操作', cls: 'acts', get: j => UI.el('span', {},
          UI.el('button', { class: 'btn sm', onclick: () => this.openJob(j.id) }, '查看'),
          j.status === 'running' ? UI.el('button', { class: 'btn sm danger', onclick: () => this.stopJob(j.id) }, '停止') : null,
          UI.el('button', { class: 'btn sm ghost', title: '以相同参数重跑', onclick: () => this.rerunJob(j) }, '重跑'),
          UI.el('button', { class: 'btn sm ghost', title: '修改标题', onclick: () => this.editJobTitle(j) }, '改名'),
          UI.el('button', { class: 'btn sm ghost', title: '删除', disabled: j.status === 'running', onclick: () => this.deleteJob(j) }, '删除'))
      },
    ], rows, '没有匹配的任务'));
  },
  async stopJob(id) {
    try { await API.post('/api/jobs/' + id + '/stop'); UI.toast('停止信号已发送（STOP 文件）'); this.loadJobs(); if (id === this.currentJob) this.refreshJobHeader(); }
    catch (e) { UI.err(e); }
  },
  async rerunJob(j) {
    try {
      const rec = await API.post('/api/jobs', {
        project_id: j.project_id, type: j.type,
        params: (j.spec && j.spec.params) || {}, title: (j.title || j.type) + ' (重跑)',
      });
      UI.toast('已提交重跑任务');
      this.openJob(rec.id);
    } catch (e) { UI.err(e); }
  },
  async editJobTitle(j) {
    const v = await UI.prompt('修改任务标题', [{ key: 'title', label: '标题', value: j.title || '' }]);
    if (!v) return;
    try { await API.patch('/api/jobs/' + j.id, { title: v.title }); UI.toast('标题已更新'); this.loadJobs(); if (j.id === this.currentJob) this.refreshJobHeader(); }
    catch (e) { UI.err(e); }
  },
  async deleteJob(j) {
    if (!await UI.confirm('删除任务', `删除 ${j.id} 及其全部日志/指标/产物？不可恢复。`, '删除')) return;
    try { await API.del('/api/jobs/' + j.id); UI.toast('任务已删除'); if (j.id === this.currentJob) { this.currentJob = null; this.showPage('jobs'); } else this.loadJobs(); }
    catch (e) { UI.err(e); }
  },

  // ------------------------------------------------------------ projects
  async loadProjects() {
    try {
      this.projects = await API.get('/api/projects');
      const sel = document.getElementById('f-project');
      const cur = sel.value;
      sel.innerHTML = '<option value="">全部项目</option>';
      const nsel = document.getElementById('new-project');
      const ncur = nsel.value;
      nsel.innerHTML = '';
      for (const p of this.projects) {
        sel.append(UI.el('option', { value: p.id }, p.name || p.id));
        nsel.append(UI.el('option', { value: p.id }, `${p.name || p.id} (${p.kind})`));
      }
      sel.value = cur; nsel.value = ncur || (this.projects[0] && this.projects[0].id) || '';
      if (this.page === 'projects') this.renderProjects();
      if (this.page === 'new') this.loadSchema();
      this.loadDeploySchema();
    } catch (e) { UI.err(e); }
  },
  renderProjects() {
    document.getElementById('proj-count').textContent = `${this.projects.length} 个项目`;
    const box = document.getElementById('proj-table');
    box.innerHTML = '';
    box.append(UI.table([
      { label: 'ID', get: p => UI.el('code', {}, p.id) },
      { label: '名称', get: p => UI.el('b', {}, p.name || p.id) },
      { label: '类型', get: p => UI.el('span', { class: 'tag' }, p.kind) },
      { label: '根目录', get: p => UI.el('span', { class: 'mono dim', style: { fontSize: '11px' } }, p.root) },
      { label: '注册', get: p => UI.el('span', { class: 'dim' }, UI.rel(p.created)) },
      {
        label: '操作', cls: 'acts', get: p => UI.el('span', {},
          UI.el('button', { class: 'btn sm', onclick: () => { document.getElementById('new-project').value = p.id; this.showPage('new'); } }, '新建任务'),
          UI.el('button', { class: 'btn sm ghost', onclick: () => this.projectModal(p) }, '编辑'),
          UI.el('button', { class: 'btn sm ghost', onclick: () => this.deleteProject(p) }, '删除'))
      },
    ], this.projects, '尚未注册项目'));
  },
  projectModal(p) {
    const edit = !!p;
    const idI = UI.el('input', { class: 'input', value: p?.id || '', placeholder: '留空自动生成', disabled: edit });
    const nameI = UI.el('input', { class: 'input', value: p?.name || '', placeholder: '显示名称' });
    const kindS = UI.el('select', { class: 'input' },
      UI.el('option', { value: 'metamodel' }, 'metamodel（TensorN MetaModel）'),
      UI.el('option', { value: 'pytorch' }, 'pytorch（PyTorch 训练项目）'),
      UI.el('option', { value: 'generic' }, 'generic（通用 CMake 项目）'));
    kindS.value = p?.kind || 'metamodel';
    const rootI = UI.el('input', { class: 'input', value: p?.root || '', placeholder: 'D:/path/to/project' });
    UI.modal({
      title: edit ? '编辑项目' : '注册项目',
      body: [
        UI.el('label', { class: 'fld' }, UI.el('span', {}, '项目 ID'), idI),
        UI.el('label', { class: 'fld' }, UI.el('span', {}, '名称'), nameI),
        UI.el('label', { class: 'fld' }, UI.el('span', {}, '类型'), kindS),
        UI.el('label', { class: 'fld' }, UI.el('span', {}, '根目录', UI.el('span', { class: 'help', title: 'metamodel：包含 build/ 与 datasets/ 的项目根；pytorch：含训练脚本的仓库根' }, '?')), rootI),
      ],
      actions: [
        { label: '取消', kind: 'ghost', onClick: c => c() },
        {
          label: edit ? '保存' : '注册', kind: 'primary', onClick: async c => {
            const body = { name: nameI.value.trim(), kind: kindS.value, root: rootI.value.trim() };
            if (!edit && idI.value.trim()) body.id = idI.value.trim();
            try {
              if (edit) await API.put('/api/projects/' + p.id, body);
              else await API.post('/api/projects', body);
              c(); UI.toast(edit ? '项目已更新' : '项目已注册');
              this.loadProjects();
            } catch (e) { UI.err(e); }
          },
        },
      ],
    });
  },
  async deleteProject(p) {
    if (!await UI.confirm('删除项目', `移除项目 ${p.id}？已提交的任务记录保留。`, '删除')) return;
    try { await API.del('/api/projects/' + p.id); UI.toast('项目已删除'); this.loadProjects(); }
    catch (e) { UI.err(e); }
  },

  // ------------------------------------------------------------ new job
  loadNew() {
    const seg = document.getElementById('new-type');
    seg.innerHTML = '';
    for (const t of ['train', 'eval', 'generate', 'test', 'build', 'docker', 'deploy', 'custom'])
      seg.append(UI.el('button', { class: t === this.newType ? 'active' : '', onclick: () => { this.newType = t; this.loadNew(); } }, t));
    this.loadSchema();
  },
  async loadSchema() {
    const pid = document.getElementById('new-project').value;
    const box = document.getElementById('new-form');
    if (!pid) { box.innerHTML = ''; box.append(UI.el('div', { class: 'empty' }, UI.el('b', {}, '先注册项目'), '在「项目」页注册后端项目根目录')); this.newForm = null; return; }
    try {
      const r = await API.get(`/api/projects/${pid}/schema?type=${this.newType}`);
      this.newForm = Form.render(box, r.schema, this.newValues[this.newType] || {});
    } catch (e) { UI.err(e); }
  },
  async submitNew() {
    const pid = document.getElementById('new-project').value;
    if (!pid) return UI.toast('请先选择项目', 'err');
    if (!this.newForm) return UI.toast('参数表单未加载', 'err');
    const params = this.newForm.collect();
    if (!params) return;
    this.newValues[this.newType] = params;
    const title = document.getElementById('new-title').value.trim();
    try {
      const rec = await API.post('/api/jobs', { project_id: pid, type: this.newType, params, title: title || undefined });
      UI.toast('任务已提交：' + rec.id);
      this.openJob(rec.id);
    } catch (e) { UI.err(e); }
  },

  // ------------------------------------------------------------ docker / deploy
  async loadDocker() {
    try {
      const [ps, im] = await Promise.all([API.get('/api/docker/ps'), API.get('/api/docker/images')]);
      const pill = document.getElementById('docker-avail');
      pill.className = 'pill ' + (ps.available ? 'on' : 'off');
      pill.lastChild.textContent = ps.available ? 'docker 可用' : 'docker 不可用';
      this.renderContainers(ps);
      this.renderImages(im);
    } catch (e) { UI.err(e); }
  },
  renderContainers(ps) {
    const box = document.getElementById('dk-containers');
    box.innerHTML = '';
    const rows = ps.rows || [];
    box.append(UI.table([
      { label: '名称', get: c => UI.el('b', { class: 'mono' }, c.Names) },
      { label: '镜像', get: c => UI.el('span', { class: 'muted mono', style: { fontSize: '11px' } }, c.Image) },
      { label: '状态', get: c => UI.el('span', { class: 'tag', style: { color: /Up/.test(c.Status || '') ? 'var(--green)' : 'var(--muted)' } }, c.Status) },
      { label: '端口', get: c => UI.el('span', { class: 'dim mono', style: { fontSize: '11px' } }, c.Ports || '—') },
      {
        label: '操作', cls: 'acts', get: c => {
          const up = /Up/.test(c.Status || '');
          return UI.el('span', {},
            up ? UI.el('button', { class: 'btn sm', onclick: () => this.dockerAction('stop', c.Names, c.Names) }, '停止')
              : UI.el('button', { class: 'btn sm', onclick: () => this.dockerAction('start', c.Names, c.Names) }, '启动'),
            up ? UI.el('button', { class: 'btn sm ghost', onclick: () => this.dockerAction('restart', c.Names, c.Names) }, '重启') : null,
            UI.el('button', { class: 'btn sm danger', onclick: () => this.dockerAction('rm', c.Names, c.Names, true) }, '删除'));
        }
      },
    ], rows, ps.available ? '没有容器' : 'docker CLI 不可用'));
  },
  renderImages(im) {
    const box = document.getElementById('dk-images');
    box.innerHTML = '';
    box.append(UI.table([
      { label: '仓库:标签', get: i => UI.el('span', { class: 'mono' }, `${i.Repository}:${i.Tag}`) },
      { label: '大小', num: true, get: i => i.Size },
      { label: 'ID', get: i => UI.el('span', { class: 'mono dim' }, (i.ID || '').slice(0, 12)) },
      { label: '创建', get: i => UI.el('span', { class: 'dim' }, i.CreatedSince || '') },
      { label: '操作', cls: 'acts', get: i => UI.el('button', { class: 'btn sm danger', onclick: () => this.dockerAction('rmi', i.ID, `${i.Repository}:${i.Tag}`, true) }, '删除') },
    ], im.rows || [], im.available ? '没有镜像' : 'docker CLI 不可用'));
  },
  async dockerAction(action, id, label, destructive = false) {
    if (destructive && !await UI.confirm('确认操作', `docker ${action} ${label}？不可恢复。`, '执行')) return;
    try {
      const r = await API.post('/api/docker/action', { action, id });
      if (r.ok) UI.toast(`docker ${action} 完成`);
      else UI.toast(`docker ${action} 失败：${r.output || 'exit ' + r.exit_code}`, 'err', 6000);
      this.loadDocker();
    } catch (e) { UI.err(e); }
  },
  async loadDeploySchema() {
    const pid = this.projects[0] && this.projects[0].id;
    if (!pid) return;
    try {
      const r = await API.get(`/api/projects/${pid}/schema?type=deploy`);
      this.deployForm = Form.render(document.getElementById('deploy-form'), r.schema, {});
    } catch { /* schema optional */ }
  },
  async submitDeploy() {
    if (!this.deployForm) return UI.toast('部署表单未加载（先注册项目）', 'err');
    const params = this.deployForm.collect();
    if (!params) return;
    try {
      const rec = await API.post('/api/deploy', { project_id: this.projects[0].id, params });
      UI.toast('部署任务已提交');
      this.openJob(rec.id);
    } catch (e) { UI.err(e); }
  },

  // ------------------------------------------------------------ job detail
  async openJob(id) {
    this.currentJob = id;
    this.metricsRows = [];
    Charts.destroyIn(document.getElementById('pane-metrics'));
    if (Moe._chart) { Moe._chart.destroy(); Moe._chart = null; }
    Moe.cache = null; Moe.sel = null; Moe.selExpert = null;
    document.getElementById('job-nav').hidden = false;
    try {
      this.job = await API.get('/api/jobs/' + id);
      this.renderJobHeader();
      this.showPage('job', this.tab || 'metrics');
    } catch (e) { UI.err(e); }
  },
  async refreshJobHeader() {
    if (!this.currentJob) return;
    try { this.job = await API.get('/api/jobs/' + this.currentJob); this.renderJobHeader(); } catch { }
  },
  renderJobHeader() {
    const j = this.job;
    if (!j) return;
    const box = document.getElementById('job-header');
    box.innerHTML = '';
    box.append(
      UI.el('h2', {}, j.title || j.id,
        UI.el('span', { class: 'edit', title: '修改标题', onclick: () => this.editJobTitle(j) }, '✎')),
      UI.badge(j.status),
      UI.el('span', { class: 'tag' }, j.type),
      UI.el('span', { class: 'kv-chips' },
        UI.el('span', { class: 'tag' }, j.project_id),
        UI.el('span', { class: 'mono dim', style: { fontSize: '11px' } }, j.id),
        (j.ranks || []).map(r => UI.el('span', { class: 'tag' }, `r${r.rank}·GPU${r.gpu}`)),
        UI.el('span', { class: 'dim', style: { fontSize: '11.5px' } }, `开始 ${UI.rel(j.started)} · 耗时 ${UI.dur(j.started, j.ended)}`),
        j.exit_code !== null && j.exit_code !== undefined ? UI.el('span', { class: 'tag', style: { color: j.exit_code === 0 ? 'var(--green)' : 'var(--red)' } }, 'exit ' + j.exit_code) : null),
      UI.el('span', { class: 'spacer', style: { flex: 1 } }),
      ...(j.status === 'running' ? [UI.el('button', { class: 'btn sm danger', onclick: () => this.stopJob(j.id) }, '停止')] : []),
      UI.el('button', { class: 'btn sm', onclick: () => this.rerunJob(j) }, '重跑'),
      UI.el('button', { class: 'btn sm ghost', onclick: () => this.refreshJobHeader() }, '刷新'),
      UI.el('button', { class: 'btn sm ghost', disabled: j.status === 'running', onclick: () => this.deleteJob(j) }, '删除'));
    document.getElementById('crumb').innerHTML = `任务详情 <span>/ ${UI.esc(j.title || j.id)}</span>`;
  },
  switchTab(tab, force = false) {
    this.tab = tab;
    if (this.page === 'job' && this.currentJob)
      this.setHash(`job/${this.currentJob}/${tab}` + (tab === 'grad' ? '/' + Osc.subview : ''));
    document.querySelectorAll('#job-tabs a').forEach(a => a.classList.toggle('active', a.dataset.tab === tab));
    document.querySelectorAll('#job-nav a').forEach(a => a.classList.toggle('active', a.dataset.tab === tab));
    document.querySelectorAll('#page-job .tabpane').forEach(p => { p.hidden = p.id !== 'pane-' + tab; });
    if (tab !== 'metrics') Charts.destroyIn(document.getElementById('pane-metrics'));
    if (tab === 'metrics') { if (!this.metricsRows.length || force) this.loadMetrics(); else this.renderMetrics(); }
    else if (tab === 'logs') this.loadLogs(force);
    else if (tab === 'grad') this.reloadGrad();
    else if (tab === 'prof') this.reloadProfile();
    else if (tab === 'report') { this.loadReport(); this.loadArtifacts(); }
  },

  async loadMetrics() {
    try {
      this.metricsRows = await API.get('/api/jobs/' + this.currentJob + '/metrics?limit=100000');
      this.renderMetrics();
    } catch (e) { UI.err(e); }
  },
  pushMetric(d) {
    const row = { step: d.step, rank: d.rank ?? 0, t: d.t, v: d.v || d };
    const i = this.metricsRows.findIndex(r => r.step === row.step && (r.rank ?? 0) === row.rank);
    if (i >= 0) this.metricsRows[i] = { ...this.metricsRows[i], ...row, v: { ...this.metricsRows[i].v, ...row.v } };
    else this.metricsRows.push(row);
    if (this.tab === 'metrics') this._metricRender();
  },
  renderMetrics() {
    const box = document.getElementById('metrics-charts');
    Charts.destroyIn(box);
    box.innerHTML = '';
    const rows = this.metricsRows;
    if (!rows.length) {
      box.append(UI.el('div', { class: 'empty', style: { gridColumn: '1/-1' } }, UI.el('b', {}, '暂无指标'), '训练产生 step 日志后自动出现'));
      return;
    }
    const ranks = [...new Set(rows.map(r => r.rank ?? 0))].sort((a, b) => a - b);
    const steps = [...new Set(rows.map(r => r.step))].sort((a, b) => a - b);
    const known = new Set(['ce', 'aux', 'grad_norm', 'lr', 'tps', 'sec', 'pool_used', 'pool_reserved', 'pool_peak', 'pool_trim', 'driver_free', 'fm', 'nodes', 'step', 'trainable', 'params_total', 'unfrozen', 'lrc_lambda', 'drift', 'grate', 'emb_rms']);
    const extra = [...new Set(rows.flatMap(r => Object.keys(r.v || {})))].filter(k => !known.has(k));
    const groups = [
      ['损失 CE / aux', ['ce', 'aux']],
      ['梯度范数 |g|', ['grad_norm']],
      ['学习率 lr', ['lr']],
      ['LR 控制 λ', ['lrc_lambda']],
      ['漂移 drift / 梯度增长 grate', ['drift', 'grate']],
      ['tied emb RMS', ['emb_rms']],
      ['吞吐 tok/s', ['tps']],
      ['step 耗时 s', ['sec']],
      ['显存池 MiB', ['pool_used', 'pool_reserved', 'pool_peak']],
      ['引擎 fm / nodes', ['fm', 'nodes']],
    ];
    if (extra.length) groups.push(['探针自定义指标', extra.slice(0, 8)]);
    let ci = 0;
    for (const [title, keys] of groups) {
      const present = keys.filter(k => rows.some(r => r.v && typeof r.v[k] === 'number'));
      if (!present.length) continue;
      const card = UI.el('div', { class: 'card' }, UI.el('div', { class: 'card-h' }, title));
      const cbox = UI.el('div', { class: 'chart-box' });
      card.append(cbox);
      box.append(card);
      const defs = [];
      for (const k of present) for (const r of ranks)
        defs.push({ label: ranks.length > 1 ? `${k} r${r}` : k, color: Charts.COLORS[ci++ % 8], key: k, rank: r });
      const h = Charts.line(cbox, defs, { height: 210 });
      const ys = defs.map(d => steps.map(s => {
        const row = rows.find(r => r.step === s && (r.rank ?? 0) === d.rank);
        const v = row && row.v ? row.v[d.key] : undefined;
        return typeof v === 'number' ? v : null;
      }));
      h.setData(steps, ys);
    }
  },

  async loadLogs(reset = false) {
    try {
      if (reset) this.logsOffset = 0;
      const r = await API.get(`/api/jobs/${this.currentJob}/logs?offset=${reset ? 0 : this.logsOffset}&limit=2000`);
      const view = document.getElementById('logs-view');
      if (reset) view.textContent = '';
      for (const l of r.lines || []) view.append(l.line + '\n');
      this.logsOffset = r.next_offset;
      this.logsTotal = r.total;
      document.getElementById('logs-count').textContent = `${this.logsOffset}/${r.total} 行`;
      if (document.getElementById('logs-follow').checked) view.scrollTop = view.scrollHeight;
    } catch (e) { UI.err(e); }
  },
  appendLog(line) {
    const view = document.getElementById('logs-view');
    view.append(line + '\n');
    this.logsOffset++; this.logsTotal++;
    document.getElementById('logs-count').textContent = `${this.logsOffset}/${this.logsTotal} 行`;
    if (document.getElementById('logs-follow').checked) view.scrollTop = view.scrollHeight;
  },

  async reloadGrad() {
    try {
      const [order, rows] = await Promise.all([
        API.get('/api/jobs/' + this.currentJob + '/grad/order'),
        API.get('/api/jobs/' + this.currentJob + '/grad?limit=100000'),
      ]);
      this.gradOrder = order;
      Osc.setData(order, rows);
      // 架构流 / MoE 视图依赖 MetaModel 命名；通用模型（PyTorch）默认热力图。
      if (order.length) {
        const meta = order.some(o => o.name === 'emb.weight' || o.name === 'head_norm.weight' ||
          o.name.startsWith('decoder_layers.'));
        document.querySelectorAll('#grad-views button').forEach(b => {
          if (b.dataset.view === 'heat') return;
          b.disabled = !meta;
          b.title = meta ? '' : 'MetaModel 专属视图，通用模型请用权重热力图';
        });
        if (!meta) {
          document.getElementById('grad-modelinfo').textContent = `generic · ${order.length} 权重`;
          if (Osc.subview !== 'heat') Osc.setSubview('heat');
        }
      }
      document.getElementById('grad-step-label').textContent =
        Osc.steps.length ? `step ${Osc.steps[Osc.step]} (${Osc.step + 1}/${Osc.steps.length})` : '无数据';
    } catch (e) { UI.err(e); }
  },
  async reloadProfile() {
    try { Flame.setData(await API.get('/api/jobs/' + this.currentJob + '/profile')); }
    catch (e) { UI.err(e); }
  },
  async loadReport() {
    try {
      const md = await API.text('/api/jobs/' + this.currentJob + '/report?format=md');
      document.getElementById('report-view').innerHTML = UI.md(md);
    } catch (e) { UI.err(e); }
  },
  async loadArtifacts() {
    try {
      const arts = await API.get('/api/jobs/' + this.currentJob + '/artifacts');
      const box = document.getElementById('artifacts-list');
      box.innerHTML = '';
      if (!arts.length) box.append(UI.el('span', { class: 'dim' }, '暂无产物'));
      for (const a of arts)
        box.append(UI.el('a', { href: API.artifactUrl(this.currentJob, a.name), target: '_blank' },
          a.name, UI.el('span', { class: 'dim' }, UI.bytes(a.size))));
    } catch (e) { UI.err(e); }
  },
};

window.App = App;
document.addEventListener('DOMContentLoaded', () => App.init());