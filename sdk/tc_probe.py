"""tc_probe.py — TensorConsole L1 probe SDK for PyTorch (Python 3.8+, stdlib only).

tcd launches ``kind=pytorch`` train jobs with TC_JOB_DIR / TC_RANK / TC_STOP_FILE
set and this directory injected into PYTHONPATH, so a training script only needs:

    from tc_probe import Probe

    probe = Probe()                 # no-op unless TC_JOB_DIR is set
    probe.watch(model)              # output->input grad order + param tracking

    for step, batch in enumerate(loader):
        loss = model(**batch).loss
        loss.backward()
        probe.step(step)            # p.grad L2 -> 梯度示波器
        probe.metrics(step, {"loss": loss, "lr": lr})
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if probe.should_stop():     # tcd 面板停止时写 <out>.STOP
            break

    with probe.scope("forward"):    # 可选：耗时 span -> trace.json 火焰图
        out = model(x)

    probe.flush()                   # 进程退出时也会自动 flush

Files written into the tcd job directory (formats tcd already ingests):

    grad_raw_r<rank>.csv   step,name,grad_l2
    grad_order.json        [{"name","module","depth"}]（默认 reversed(named_parameters)）
    metrics.jsonl          {"step","t","rank","v":{...},"src":"probe"}
    trace.json             Chrome Trace Event（ph=X）

Every call is a cheap no-op when TC_JOB_DIR is absent, so the same script runs
standalone unchanged.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from contextlib import contextmanager

__all__ = ["Probe"]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_us() -> int:
    return time.perf_counter_ns() // 1000


def _detect_rank() -> int:
    # torchrun sets LOCAL_RANK for workers; tcd only sets TC_RANK (parent).
    for key in ("LOCAL_RANK", "RANK", "TC_RANK"):
        raw = os.environ.get(key)
        if not raw:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return 0


def _num(v):
    """Best-effort numeric conversion (torch/numpy scalars) for chart series."""
    try:
        if hasattr(v, "item"):
            v = v.item()
        return float(v)
    except Exception:
        return v


class Probe:
    def __init__(self, job_dir=None, rank=None):
        self.dir = job_dir if job_dir is not None else os.environ.get("TC_JOB_DIR", "")
        self.enabled = bool(self.dir)
        self.rank = _detect_rank() if rank is None else int(rank)
        self.stop_file = os.environ.get("TC_STOP_FILE", "")
        self._grad_path = os.environ.get("TC_PROBE_GRAD_FILE") or (
            os.path.join(self.dir, "grad_raw_r%d.csv" % self.rank) if self.dir else ""
        )
        self._cuda_sync = os.environ.get("TC_PROBE_CUDA_SYNC", "") == "1"
        self._lock = threading.Lock()
        self._params = []   # [(full_name, tensor)]
        self._order = []    # output -> input parameter names
        self._events = []   # Chrome Trace events (JSON strings)
        self._local = threading.local()
        if self.enabled:
            atexit.register(self.flush)

    # ------------------------------------------------------------- metrics

    def metrics(self, step, kv=None, **kwargs):
        """Log per-step scalars:
        probe.metrics(step, {"loss": loss, "lr": lr}) 或 probe.metrics(step, loss=loss).
        """
        if not self.enabled:
            return
        values = {}
        if kv:
            values.update(kv)
        values.update(kwargs)
        if not values:
            return
        row = {
            "step": int(step),
            "t": _now_ms(),
            "rank": self.rank,
            "v": {k: _num(v) for k, v in values.items()},
            "src": "probe",
        }
        self._append("metrics.jsonl", json.dumps(row, separators=(",", ":"), default=str) + "\n")

    # ------------------------------------------------------------- gradients

    def watch(self, model, prefix="", order="reversed"):
        """Track model.named_parameters() and declare the grad order.

        order: "reversed" (default, output->input for feed-forward stacks),
               "named" (named_parameters order) or an explicit iterable of names.
        Each model parameter name should appear once; DDP prefixes are kept.
        """
        if not self.enabled:
            return self
        self._params = [(prefix + n, p) for n, p in model.named_parameters()]
        names = [n for n, _ in self._params]
        if order == "reversed":
            names = list(reversed(names))
        elif order == "named":
            pass
        elif order is not None:
            names = [str(n) for n in order]
        self._order = names
        if self.rank == 0:  # torchrun ranks share one job dir
            entries = []
            for depth, name in enumerate(names):
                dot = name.find(".")
                module = name[:dot] if dot > 0 else name
                entries.append({"name": name, "module": module, "depth": depth})
            self._write_json("grad_order.json", entries)
        return self

    def step(self, step):
        """Emit L2 norms of tracked params that currently hold .grad.

        Call right after loss.backward() and before optimizer.zero_grad().
        """
        if not (self.enabled and self._params):
            return
        by_name = dict(self._params)
        rows = []
        for name in self._order or list(by_name):
            p = by_name.get(name)
            if p is None:
                continue
            g = getattr(p, "grad", None)
            if g is None:
                continue
            try:
                l2 = float(g.detach().norm(2).item())
            except Exception:
                continue
            rows.append("%d,%s,%.9g\n" % (int(step), name, l2))
        if rows:
            self._append_file(self._grad_path, "".join(rows))

    def grad(self, step, name, l2):
        """Manual single-weight gradient sample (non-torch pipelines)."""
        if not self.enabled:
            return
        self._append_file(self._grad_path, "%d,%s,%.9g\n" % (int(step), name, float(l2)))

    # ------------------------------------------------------------- timing

    def _stack(self):
        st = getattr(self._local, "stack", None)
        if st is None:
            st = self._local.stack = []
        return st

    def _maybe_sync(self):
        if not self._cuda_sync:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def begin(self, name):
        if not self.enabled:
            return
        self._maybe_sync()
        self._stack().append((str(name), _now_us()))

    def end(self):
        if not self.enabled:
            return
        self._maybe_sync()
        st = self._stack()
        if not st:
            return
        name, ts = st.pop()
        ev = {
            "name": name,
            "cat": "probe",
            "ph": "X",
            "ts": ts,
            "dur": _now_us() - ts,
            "pid": os.getpid(),
            "tid": threading.get_ident(),
        }
        with self._lock:
            self._events.append(json.dumps(ev, separators=(",", ":")))

    @contextmanager
    def scope(self, name):
        self.begin(name)
        try:
            yield self
        finally:
            self.end()

    def flush(self):
        """Write the trace file (no-op without recorded spans). Also runs at exit.

        Rank 0 writes trace.json, other DDP ranks write trace_r<rank>.json;
        tcd merges them into one flame graph when the job finalizes.
        """
        if not self.enabled:
            return
        with self._lock:
            events = list(self._events)
        if not events:
            return
        trace = {"displayTimeUnit": "us", "traceEvents": [json.loads(e) for e in events]}
        name = "trace.json" if self.rank == 0 else "trace_r%d.json" % self.rank
        self._write_json(name, trace)

    # ------------------------------------------------------------- control

    def should_stop(self):
        """True once tcd's stop button wrote the <out>.STOP file."""
        return bool(self.stop_file) and os.path.exists(self.stop_file)

    # ------------------------------------------------------------- io

    def _append(self, name, text):
        self._append_file(os.path.join(self.dir, name), text)

    def _append_file(self, path, text):
        if not path:
            return
        # One os.write keeps multi-rank appends of metrics.jsonl atomic
        # (buffered text-mode append loses lines on Windows).
        try:
            data = text.encode("utf-8")
            with self._lock:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND
                             | getattr(os, "O_BINARY", 0), 0o666)
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
        except OSError:
            pass  # a probe must never break training

    def _write_json(self, name, obj):
        path = os.path.join(self.dir, name)
        tmp = "%s.tmp.%d" % (path, os.getpid())
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, separators=(",", ":"))
            os.replace(tmp, path)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass