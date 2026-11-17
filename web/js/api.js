// TensorConsole API client + SSE
const API = {
  token: localStorage.getItem('tc_token') || '',
  base: '',
  setToken(t) { this.token = t || ''; localStorage.setItem('tc_token', this.token); },
  headers(json) {
    const h = {};
    if (this.token) h['Authorization'] = 'Bearer ' + this.token;
    if (json) h['Content-Type'] = 'application/json';
    return h;
  },
  async req(method, path, body) {
    const opt = { method, headers: this.headers(body !== undefined) };
    if (body !== undefined) opt.body = JSON.stringify(body);
    const r = await fetch(this.base + path, opt);
    const txt = await r.text();
    let j = null;
    try { j = txt ? JSON.parse(txt) : null; } catch { j = { raw: txt }; }
    if (!r.ok) {
      const msg = (j && (j.error || j.message)) || txt || r.statusText;
      const e = new Error(msg); e.status = r.status; throw e;
    }
    return j;
  },
  get(p) { return this.req('GET', p); },
  post(p, b) { return this.req('POST', p, b === undefined ? {} : b); },
  put(p, b) { return this.req('PUT', p, b); },
  patch(p, b) { return this.req('PATCH', p, b); },
  del(p) { return this.req('DELETE', p); },
  async text(p) {
    const r = await fetch(this.base + p, { headers: this.headers(false) });
    if (!r.ok) throw new Error((await r.text()) || r.statusText);
    return r.text();
  },
  artifactUrl(id, name) {
    let u = this.base + '/api/jobs/' + encodeURIComponent(id) + '/artifacts/' + encodeURIComponent(name);
    if (this.token) u += '?token=' + encodeURIComponent(this.token);
    return u;
  },
  // SSE with auto-reconnect; handlers: {event: fn}, onState(ok)
  events(topics, handlers, onState) {
    let es = null, closed = false, retry = 0;
    const open = () => {
      if (closed) return;
      let u = this.base + '/api/events';
      const q = [];
      if (topics) q.push('topics=' + encodeURIComponent(topics));
      if (this.token) q.push('token=' + encodeURIComponent(this.token));
      if (q.length) u += '?' + q.join('&');
      es = new EventSource(u);
      es.onopen = () => { retry = 0; onState && onState(true); };
      es.onerror = () => {
        onState && onState(false);
        es.close();
        if (!closed) setTimeout(open, Math.min(8000, 800 * ++retry));
      };
      for (const [name, fn] of Object.entries(handlers))
        es.addEventListener(name, ev => {
          let d = null;
          try { d = JSON.parse(ev.data); } catch { d = ev.data; }
          fn(d, name);
        });
    };
    open();
    return { close() { closed = true; es && es.close(); } };
  },
};