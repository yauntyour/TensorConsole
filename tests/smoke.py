#!/usr/bin/env python3
"""TensorConsole end-to-end smoke test (zero deps, stdlib only).

Usage:
    python tests/smoke.py [--server http://127.0.0.1:8686] [--token T]
                          [--project-root D:/Developments/ML/CXX-Backend/MetaModel]
                          [--skip-stop] [--skip-multirank]

Requires a running tcd and a MetaModel checkout with build/train.exe.
Exits 0 when all checks pass, 1 otherwise.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


class Client:
    def __init__(self, server, token):
        self.base = server.rstrip("/")
        self.token = token

    def req(self, method, path, body=None, raw=False, timeout=30):
        url = self.base + path
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        r = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                payload = resp.read()
                return resp.status, payload if raw else json.loads(payload or b"{}")
        except urllib.error.HTTPError as e:
            payload = e.read()
            try:
                return e.code, json.loads(payload)
            except Exception:
                return e.code, {"error": payload.decode(errors="replace")}

    def wait_terminal(self, jid, timeout_s=300):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            _, j = self.req("GET", f"/api/jobs/{jid}")
            if j.get("status") in ("succeeded", "failed", "stopped"):
                return j
            time.sleep(1.0)
        return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8686")
    ap.add_argument("--token", default="")
    ap.add_argument("--project-root", default="D:/Developments/ML/CXX-Backend/MetaModel")
    ap.add_argument("--skip-stop", action="store_true")
    ap.add_argument("--skip-multirank", action="store_true")
    a = ap.parse_args()
    c = Client(a.server, a.token)

    print("[1] health / system / gpus")
    st, h = c.req("GET", "/api/health")
    check("health 200 ok", st == 200 and h.get("ok"), str(h))
    st, sysinfo = c.req("GET", "/api/system")
    check("system hostname", st == 200 and bool(sysinfo.get("hostname")))
    st, gpus = c.req("GET", "/api/gpus")
    check("gpus endpoint", st == 200)

    print("[2] project + schema")
    st, projects = c.req("GET", "/api/projects")
    pid = "smoke-metamodel"
    if not any(p.get("id") == pid for p in projects):
        st, p = c.req("POST", "/api/projects",
                      {"id": pid, "name": "Smoke MetaModel", "kind": "metamodel",
                       "root": a.project_root})
        check("project create", st in (200, 201), str(p))
    st, schema = c.req("GET", f"/api/projects/{pid}/schema?type=train")
    groups = schema.get("schema", {}).get("groups", [])
    keys = {f["key"] for g in groups for f in g["fields"]}
    check("schema train groups", len(groups) >= 9, f"got {len(groups)}")
    check("schema key coverage",
          {"d_model", "n_experts", "lr", "gpus", "grad", "prof", "dtype"} <= keys)

    print("[3] train submit + wait")
    params = {"scratch": True, "steps": 8, "seq_len": 512, "batch": 1, "grad_accum": 1,
              "d_model": 256, "n_heads": 4, "n_decoder": 4, "n_groups": 4, "n_experts": 4,
              "dtype": 0, "grad": True, "prof": True, "log_interval": 1,
              "eval_interval": 1000, "save_interval": 1000, "lr": 1e-4}
    st, rec = c.req("POST", "/api/jobs",
                    {"project_id": pid, "type": "train", "params": params})
    check("submit 201", st == 201 and rec.get("id"), str(rec))
    jid = rec["id"]
    j = c.wait_terminal(jid)
    check("train succeeded", j.get("status") == "succeeded" and j.get("exit_code") == 0,
          f"status={j.get('status')} exit={j.get('exit_code')}")

    print("[4] metrics / grad / profile / report")
    st, m = c.req("GET", f"/api/jobs/{jid}/metrics")
    rows = m if isinstance(m, list) else m.get("rows", [])
    check("metrics rows", len(rows) >= 8, f"got {len(rows)}")
    check("metrics ce/grad_norm/tps",
          all(any(k in r.get("v", {}) for r in rows) for k in ("ce", "grad_norm", "tps")))
    st, order = c.req("GET", f"/api/jobs/{jid}/grad/order")
    names = [e["name"] for e in order]
    check("grad order unique", len(names) == len(set(names)),
          f"{len(names)} vs {len(set(names))}")
    check("grad order output->input",
          names and names[0] == "head_norm.weight" and names[-1] == "emb.weight",
          f"first={names[:1]} last={names[-1:]}")
    st, g = c.req("GET", f"/api/jobs/{jid}/grad")
    grows = g if isinstance(g, list) else g.get("rows", [])
    check("grad rows per step", len(grows) == 8
          and all(len(r.get("norms", [])) == len(names) for r in grows),
          f"got {len(grows)} rows")
    st, prof = c.req("GET", f"/api/jobs/{jid}/profile")
    check("profile eng_prof", prof.get("source") == "eng_prof" and prof.get("summary"),
          str(prof.get("source")))
    st, rep = c.req("GET", f"/api/jobs/{jid}/report?format=md", raw=True)
    check("report md", st == 200 and jid.encode() in rep)
    st, arts = c.req("GET", f"/api/jobs/{jid}/artifacts")
    anames = {x["name"] for x in arts}
    check("artifacts", {"report.md", "profile.json", "grad_order.json"} <= anames,
          str(anames))

    print("[5] chrome trace upload")
    trace = {"traceEvents": [
        {"ph": "X", "name": "step", "cat": "train", "pid": 1, "tid": 1, "ts": 0, "dur": 10000},
        {"ph": "X", "name": "fwd", "cat": "op", "pid": 1, "tid": 1, "ts": 100, "dur": 4000},
        {"ph": "X", "name": "bwd", "cat": "op", "pid": 1, "tid": 1, "ts": 4200, "dur": 5000},
    ]}
    st, _ = c.req("POST", f"/api/jobs/{jid}/profile", trace)
    check("trace upload 200", st == 200)
    st, prof2 = c.req("GET", f"/api/jobs/{jid}/profile")
    check("chrome_trace parsed", prof2.get("source") == "chrome_trace"
          and len(prof2.get("spans", [])) == 3
          and {s["depth"] for s in prof2["spans"]} == {0, 1})

    if not a.skip_stop:
        print("[6] stop semantics")
        lp = dict(params, steps=100000, prof=False, log_interval=20)
        st, rec2 = c.req("POST", "/api/jobs",
                         {"project_id": pid, "type": "train", "params": lp})
        jid2 = rec2["id"]
        t0 = time.time()
        running = False
        while time.time() - t0 < 60:
            _, jj = c.req("GET", f"/api/jobs/{jid2}")
            if jj.get("status") == "running":
                running = True
                break
            time.sleep(0.5)
        check("long job running", running)
        time.sleep(8)
        st, _ = c.req("POST", f"/api/jobs/{jid2}/stop")
        check("stop accepted", st == 200)
        j2 = c.wait_terminal(jid2, timeout_s=60)
        check("stopped terminal", j2.get("status") == "stopped",
              f"status={j2.get('status')}")

    if not a.skip_multirank:
        print("[7] multi-rank (2 ranks on GPU 0)")
        mp = dict(params, steps=6, gpus="0,0", data_shard=True, prof=False)
        st, rec3 = c.req("POST", "/api/jobs",
                         {"project_id": pid, "type": "train", "params": mp})
        jid3 = rec3["id"]
        check("2 ranks spawned", len(rec3.get("ranks", [])) == 2)
        j3 = c.wait_terminal(jid3)
        check("multirank succeeded", j3.get("status") == "succeeded",
              f"status={j3.get('status')}")
        st, g3 = c.req("GET", f"/api/jobs/{jid3}/grad")
        g3rows = g3 if isinstance(g3, list) else g3.get("rows", [])
        check("per-rank grad", {r.get("rank") for r in g3rows} == {0, 1}
              and len(g3rows) == 6 * 2
              and all(len(r.get("norms", [])) == len(names) for r in g3rows),
              f"got {len(g3rows)}")

    print(f"\nresult: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())