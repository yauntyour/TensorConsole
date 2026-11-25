// uPlot wrappers: container-fitted, resize-aware, correct series binding
const Charts = {
  _all: new Set(),
  COLORS: ['#2dd4bf', '#60a5fa', '#a78bfa', '#fbbf24', '#f472b6', '#34d399', '#fb923c', '#e879f9'],

  // step = spacing between adjacent ticks; without it a near-constant series
  // produces ticks that all round to the same label (e.g. 10.02 / 10.04 -> "10.0")
  tick(v, step) {
    if (v === null || v === undefined || isNaN(v)) return '';
    if (step > 0 && step < 1e-4) return v.toExponential(1);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-4 || a >= 1e6)) return v.toExponential(1);
    const d = step > 0 ? Math.max(0, Math.min(6, Math.ceil(-Math.log10(step))))
      : a >= 100 ? 0 : a >= 1 ? 1 : 3;
    return v.toFixed(d);
  },

  // defs: [{label, color, scale?}] ; returns handle {setData(xs, ys[]), destroy}
  line(box, defs, { height = 200, yUnit = '', xTime = false, legend = true } = {}) {
    box.innerHTML = '';
    const wrap = UI.el('div', { class: 'chart-fit' });
    box.append(wrap);
    const scales = [...new Set(defs.map(d => d.scale || 'y'))];
    const yAxis = (s, i) => ({
      scale: s, side: i === 0 ? 3 : 1,
      stroke: TC.dim, grid: { stroke: '#1b242e', show: i === 0 }, ticks: { stroke: '#1b242e', show: i === 0 },
      size: s === 'pct' ? 44 : yUnit === 'MiB' ? 78 : 52, font: '10px ui-monospace, Consolas, monospace',
      values: (u, vs) => {
        const step = vs.length > 1 ? Math.abs(vs[1] - vs[0]) : 0;
        return vs.map(v => (s === 'pct' ? Charts.tick(v, step || 10) + '%' : Charts.tick(v, step) + (yUnit ? ' ' + yUnit : '')));
      },
      space: 22,
    });
    const opts = {
      width: Math.max(120, wrap.clientWidth || box.clientWidth || 320),
      height,
      series: [{}, ...defs.map(d => ({
        label: d.label, stroke: d.color, width: 1.6, spanGaps: true, scale: d.scale || 'y',
        points: { show: false },
      }))],
      axes: [
        {
          stroke: TC.dim, grid: { stroke: '#1b242e' }, ticks: { stroke: '#1b242e' },
          size: 26, font: '10px ui-monospace, Consolas, monospace',
          values: (u, vs) => vs.map(v => xTime ? UI.clock(v * 1000) : UI.num(v, 0)),
          space: xTime ? 70 : 46,
        },
        ...scales.map(yAxis),
      ],
      legend: { show: legend, live: false },
      cursor: { drag: { x: true, y: false, setScale: true } },
      scales: { x: { time: xTime }, ...Object.fromEntries(scales.map(s => [s, {}])) },
    };
    const plot = new uPlot(opts, [[] , ...defs.map(() => [])], wrap);
    const h = {
      plot, box,
      setData(xs, ys) { plot.setData([xs, ...ys], true); },
      destroy() { ro.disconnect(); plot.destroy(); Charts._all.delete(h); },
    };
    const ro = new ResizeObserver(UI.debounce(() => {
      const w = wrap.clientWidth;
      if (w > 40 && Math.abs(w - plot.width) > 2) plot.setSize({ width: w, height });
    }, 80));
    ro.observe(wrap);
    Charts._all.add(h);
    return h;
  },

  destroyIn(root) {
    for (const h of [...Charts._all])
      if (!root || root.contains(h.box)) h.destroy();
  },
  destroyAll() { Charts.destroyIn(null); },
};