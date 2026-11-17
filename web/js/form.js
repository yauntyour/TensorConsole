// Schema-driven task form: collapsible groups, typed inputs, defaults as placeholders
const Form = {
  render(container, schema, values = {}) {
    container.innerHTML = '';
    const inputs = {};
    const groups = (schema && schema.groups) || [];
    if (schema && schema.positional && schema.positional.length)
      container.append(UI.el('div', { class: 'card' }, UI.el('div', { class: 'card-b', style: { color: 'var(--muted)', fontSize: '12px' } },
        '位置参数：', UI.el('code', {}, schema.positional.join(' ')), ' — 在下方同名字段填写，按顺序传递')));
    for (const g of groups) {
      const det = UI.el('details', { class: 'form-group', open: true });
      det.append(UI.el('summary', {}, g.name, UI.el('span', { class: 'cnt' }, `${g.fields.length} 项`)));
      const grid = UI.el('div', { class: 'form-grid' });
      for (const f of g.fields) {
        const v = values[f.key];
        const has = v !== undefined && v !== null && v !== '';
        let inp;
        if (f.type === 'flag') {
          inp = UI.el('input', { type: 'checkbox', 'data-key': f.key });
          inp.checked = has ? !!v : !!f.def;
          const sw = UI.el('label', { class: 'switch', style: { paddingTop: '4px' } }, inp, UI.el('i'),
            UI.el('span', { class: 'lbl' }, has || f.def ? '启用' : '关闭'));
          inp.addEventListener('change', () => { sw.lastChild.textContent = inp.checked ? '启用' : '关闭'; });
          inputs[f.key] = { type: 'flag', el: inp };
          grid.append(UI.el('label', { class: 'fld' },
            UI.el('span', {}, f.label, f.help ? UI.el('span', { class: 'help', title: f.help }, '?') : null), sw));
          continue;
        }
        if (f.type === 'enum') {
          inp = UI.el('select', { class: 'input', 'data-key': f.key },
            UI.el('option', { value: '' }, f.def !== undefined ? `默认 (${f.def})` : '默认'),
            (f.opts || []).map(o => UI.el('option', { value: String(o) }, String(o))));
          if (has) inp.value = String(v);
        } else {
          inp = UI.el('input', {
            class: 'input',
            'data-key': f.key,
            type: f.type === 'int' || f.type === 'float' ? 'number' : 'text',
            step: f.type === 'int' ? '1' : 'any',
            placeholder: f.def !== undefined && f.def !== null ? String(f.def) : '',
            value: has ? String(v) : '',
          });
        }
        inputs[f.key] = { type: f.type, el: inp };
        grid.append(UI.el('label', { class: 'fld' },
          UI.el('span', {}, f.label, UI.el('span', { class: 'help', title: f.key + (f.help ? ' — ' + f.help : '') }, '?')),
          inp,
          f.help ? UI.el('span', { class: 'hint' }, f.help) : null));
      }
      det.append(grid);
      container.append(det);
    }
    if (!groups.length)
      container.append(UI.el('div', { class: 'empty' }, UI.el('b', {}, '该类型无参数 schema'), '选择项目与类型后加载'));
    return {
      collect() {
        const out = {};
        for (const [k, { type, el }] of Object.entries(inputs)) {
          if (type === 'flag') { out[k] = el.checked; continue; }
          const s = el.value.trim();
          if (s === '') continue;
          if (type === 'int' || type === 'float') {
            const n = Number(s);
            if (!isFinite(n)) { UI.toast(`参数 ${k} 不是合法数字`, 'err'); return null; }
            out[k] = type === 'int' ? Math.round(n) : n;
          } else out[k] = s;
        }
        return out;
      },
    };
  },
};