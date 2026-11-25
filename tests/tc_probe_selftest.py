#!/usr/bin/env python3
"""Contract self-test for sdk/tc_probe.py (stdlib only; torch optional).

Verifies the exact file formats tcd ingests, without needing a running daemon:
  grad_raw_r<rank>.csv   step,name,grad_l2
  grad_order.json        [{"name","module","depth"}]
  metrics.jsonl          {"step","t","rank","v":{...},"src":"probe"}
  trace.json             Chrome Trace (ph=X)

Usage: python tests/tc_probe_selftest.py
"""
import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SDK = os.path.join(HERE, "..", "sdk")
sys.path.insert(0, os.path.abspath(SDK))

import tc_probe  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


class FakeGrad:
    def __init__(self, v):
        self.v = v

    def detach(self):
        return self

    def norm(self, p=2):
        return self

    def item(self):
        return self.v


class FakeParam:
    def __init__(self, v=None):
        self.grad = FakeGrad(v) if v is not None else None


class FakeModel:
    def __init__(self):
        self.params = [
            ("emb.weight", FakeParam(0.5)),
            ("layers.0.attn.qkv.weight", FakeParam(1.5)),
            ("layers.0.mlp.down.weight", FakeParam(2.5)),
            ("head.weight", FakeParam(None)),  # frozen / no grad this step
        ]

    def named_parameters(self):
        return iter(self.params)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_disabled():
    print("[1] disabled without TC_JOB_DIR")
    for k in ("TC_JOB_DIR", "TC_STOP_FILE", "TC_RANK", "LOCAL_RANK"):
        os.environ.pop(k, None)
    probe = tc_probe.Probe()
    model = FakeModel()
    probe.watch(model)
    probe.step(0)
    probe.metrics(0, {"loss": 1.0})
    with probe.scope("noop"):
        pass
    probe.flush()
    check("enabled=False", probe.enabled is False)
    check("should_stop False", probe.should_stop() is False)


def test_files():
    print("[2] enabled: probe file formats")
    with tempfile.TemporaryDirectory() as td:
        stop = os.path.join(td, "run.STOP")
        os.environ["TC_JOB_DIR"] = td
        os.environ["TC_STOP_FILE"] = stop
        os.environ["TC_RANK"] = "0"
        os.environ.pop("LOCAL_RANK", None)
        os.environ.pop("RANK", None)

        probe = tc_probe.Probe()
        check("enabled=True", probe.enabled is True)
        check("rank=0", probe.rank == 0)

        model = FakeModel()
        probe.watch(model)
        probe.step(7)
        probe.grad(7, "manual.weight", 4.0)
        probe.metrics(7, {"loss": FakeGrad(1.25)}, lr=3e-4)
        with probe.scope("fwd"):
            pass
        probe.flush()

        order = json.loads(read(os.path.join(td, "grad_order.json")))
        names = [e["name"] for e in order]
        check("grad_order reversed", names == [
            "head.weight", "layers.0.mlp.down.weight",
            "layers.0.attn.qkv.weight", "emb.weight"], str(names))
        check("grad_order depth", [e["depth"] for e in order] == [0, 1, 2, 3])

        lines = [l for l in read(os.path.join(td, "grad_raw_r0.csv")).splitlines() if l]
        check("grad rows = tracked-with-grad + manual", len(lines) == 4, str(lines))
        check("grad rows step=7", all(l.startswith("7,") for l in lines), str(lines))
        check("grad rows reversed order",
              [l.split(",")[1] for l in lines[:3]] ==
              ["layers.0.mlp.down.weight", "layers.0.attn.qkv.weight", "emb.weight"],
              str(lines))
        check("grad row l2", abs(float(lines[0].split(",")[2]) - 2.5) < 1e-9)

        rows = [json.loads(l) for l in read(os.path.join(td, "metrics.jsonl")).splitlines() if l]
        check("metrics row", len(rows) == 1 and rows[0]["src"] == "probe"
              and rows[0]["step"] == 7 and rows[0]["rank"] == 0)
        check("metrics values", abs(rows[0]["v"]["loss"] - 1.25) < 1e-12
              and abs(rows[0]["v"]["lr"] - 3e-4) < 1e-12, str(rows))

        trace = json.loads(read(os.path.join(td, "trace.json")))
        evs = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        check("trace span", len(evs) == 1 and evs[0]["name"] == "fwd"
              and evs[0]["dur"] >= 0)

        check("stop before", probe.should_stop() is False)
        with open(stop, "w") as f:
            f.write("stop\n")
        check("stop after", probe.should_stop() is True)


def test_rank():
    print("[3] rank detection (torchrun LOCAL_RANK wins)")
    with tempfile.TemporaryDirectory() as td:
        os.environ["TC_JOB_DIR"] = td
        os.environ["TC_RANK"] = "0"
        os.environ["LOCAL_RANK"] = "3"
        try:
            probe = tc_probe.Probe()
            check("local rank", probe.rank == 3)
            probe.grad(1, "w", 0.25)
            check("rank file", os.path.exists(os.path.join(td, "grad_raw_r3.csv")))
        finally:
            os.environ.pop("LOCAL_RANK", None)


def test_torch():
    print("[4] real torch model (optional)")
    try:
        import torch
    except ImportError:
        print("  SKIP  torch not installed")
        return
    with tempfile.TemporaryDirectory() as td:
        os.environ["TC_JOB_DIR"] = td
        os.environ.pop("LOCAL_RANK", None)
        os.environ["TC_RANK"] = "0"
        probe = tc_probe.Probe()
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
        probe.watch(model)
        x = torch.randn(3, 4)
        loss = model(x).pow(2).mean()
        loss.backward()
        probe.step(0)
        rows = [l for l in read(os.path.join(td, "grad_raw_r0.csv")).splitlines() if l]
        check("torch grad rows", len(rows) == 4, str(rows))
        check("torch grad norm > 0",
              all(float(l.split(",")[2]) > 0 for l in rows), str(rows))


def main():
    for fn in (test_disabled, test_files, test_rank, test_torch):
        try:
            fn()
        except Exception:
            global FAIL
            FAIL += 1
            print(f"  FAIL  {fn.__name__} raised:")
            traceback.print_exc()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())