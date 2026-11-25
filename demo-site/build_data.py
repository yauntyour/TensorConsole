"""Build the static replay dataset for the public TensorConsole demo.

Reads one recorded tcd job and emits compact JSON the offline frontend can load
without a running daemon. Absolute local paths are stripped before publication.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
JOB_DIR = os.path.join(ROOT, "..", "data", "jobs", "j-20261118-160615-40d7")
OUT = os.path.join(ROOT, "web", "data")


def load(name):
    with open(os.path.join(JOB_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def jsonl(name):
    with open(os.path.join(JOB_DIR, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(name, obj, indent=None):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=None if indent else (",", ":"), indent=indent)
    print(f"{name:16s} {os.path.getsize(path)/1024:9.1f} KiB")


def sig(v, p=5):
    if v == 0:
        return 0
    return float("%.*g" % (p, v))


job = load("job.json")
order = load("grad_order.json")
grad_rows = jsonl("grad.jsonl")
metric_rows = jsonl("metrics.jsonl")
with open(os.path.join(JOB_DIR, "stdout.log"), encoding="utf-8") as f:
    log_lines = [l.rstrip("\n") for l in f]

names = [o["name"] for o in order]
idx = {n: i for i, n in enumerate(names)}

# --- grad: positional numeric norms (Osc.rebuild already accepts number[]) ---
grad = []
for r in grad_rows:
    vec = [0] * len(names)
    for n, v in r["norms"]:
        if n in idx:
            vec[idx[n]] = sig(v)
    grad.append({"step": r["step"], "rank": r.get("rank", 0), "norms": vec})

# --- metrics: merge every source row per (step, rank), as the REST API does ---
merged = {}
for m in metric_rows:
    key = (m["step"], m.get("rank", 0))
    e = merged.setdefault(key, {"step": m["step"], "rank": m.get("rank", 0), "t": m.get("t"), "v": {}})
    e["v"].update(m.get("v", {}))
    if m.get("t") and (e["t"] is None or m["t"] < e["t"]):
        e["t"] = m["t"]
metrics = [merged[k] for k in sorted(merged)]

# --- job record: drop machine-specific absolute paths ---
pub_job = dict(job)
pub_job["root"] = "MetaModel"
pub_job["env"] = {k: v for k, v in job.get("env", {}).items()
                  if k in ("ENG_UNBUF", "TC_RANK", "CUDA_VISIBLE_DEVICES")}
pub_job["env"]["ENG_DUMP_GRAD"] = "grad_raw_r0.csv"
pub_job["spec"] = {"params": {k: v for k, v in job["spec"]["params"].items()}}
pub_job["demo"] = True

# --- report: mirrors profiler::make_report ---
final = {}
for m in metrics:
    final.update(m["v"])


def fmt(v):
    return "%.6g" % v


last = grad[-1]
pairs = sorted(zip(names, last["norms"]), key=lambda p: -p[1])
zeros = [n for n, v in zip(names, last["norms"]) if v == 0]
tail = "\n".join(log_lines[-40:])

md = [f"# TensorConsole 报表 — {pub_job.get('title', pub_job['id'])}", "",
      "| 项 | 值 |", "|---|---|",
      f"| Job | `{pub_job['id']}` |",
      f"| 项目 | {pub_job['project_id']} |",
      f"| 类型 | {pub_job['type']} |",
      f"| 状态 | {pub_job['status']} |",
      f"| 退出码 | {pub_job.get('exit_code', '-')} |",
      f"| 开始 | {pub_job.get('started', '')} |",
      f"| 结束 | {pub_job.get('ended', '')} |", "",
      "> 本报表由离线回放数据生成，数值与录制时 tcd 产生的报表一致。", "",
      "## 训练指标（末值）", "", "| 指标 | 值 |", "|---|---|"]
for k in sorted(final):
    if isinstance(final[k], (int, float)):
        md.append(f"| {k} | {fmt(final[k])} |")
md += ["", f"## 梯度示波器摘要（step {last['step']}）", "",
       f"参数数：{len(names)}，零梯度：{len(zeros)}", "",
       "Top-10 梯度范数：", "", "| 参数 | grad_l2 |", "|---|---|"]
for n, v in pairs[:10]:
    md.append(f"| {n} | {fmt(v)} |")
if zeros:
    md += ["", "零梯度参数（可能死支路/冻结）："]
    md += [f"- {n}" for n in zeros[:20]]
md += ["", "## 日志尾部", "", "```", tail, "```", ""]

report_json = {
    "job": pub_job,
    "final_metrics": final,
    "grad_summary": {"step": last["step"], "n_params": len(names), "zero_grad": zeros,
                     "top10": [{"name": n, "l2": v} for n, v in pairs[:10]]},
    "profile_summary": [],
}

projects = [{"id": "metamodel", "name": "MetaModel", "kind": "metamodel",
             "root": "MetaModel", "created": job["created"]}]

dump("order.json", order)
dump("grad.json", grad)
dump("metrics.json", metrics)
dump("job.json", pub_job, indent=1)
dump("jobs.json", [pub_job])
dump("projects.json", projects)
dump("logs.json", {"lines": [{"line": l} for l in log_lines],
                   "total": len(log_lines), "next_offset": len(log_lines)})
dump("report.json", {"md": "\n".join(md), "json": report_json})
dump("meta.json", {
    "job_id": pub_job["id"], "steps": len(grad), "weights": len(names),
    "layers": 10, "experts": 8, "default_step": len(grad) - 1,
    "recorded": "2026-11-18", "duration_s": 636.0,
})