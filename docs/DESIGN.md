# TensorConsole 总体设计

> 基于 TensorN C++ Backend 的 Train/Infer 平台可视化管理面板。
> 后端：C++20 + [servic.cpp](https://github.com/yauntyour/servic.cpp)（Boost.Asio 协程单头文件服务端）
> 前端：HTML5 + 原生 JS（无构建步骤，离线可用）
> 张量/训练引擎：[TensorN](https://github.com/yauntyour/TensorN) 及其上层引擎（首个适配目标：MetaModel）

---

## 1. 目标与范围

| 能力 | 说明 |
|---|---|
| 构建控制 | 远程驱动 CMake configure/build、清理、目标选择、CUDA 架构切换 |
| 训练/推理管理 | 任务提交、参数全量配置、启停、断点续训、日志/指标实时流 |
| 部署控制 | Docker 镜像构建/容器运行、Linux systemd 服务、部署脚本编排 |
| 数据可视化 | loss / tps / lr / grad_norm / 显存 / GPU 利用率等实时曲线 |
| **梯度示波器** | 按输出→输入顺序、精确到每个权重的逐 step 梯度范数可视化 |
| **全流程耗时分析** | 算子级/函数级耗时表 + 火焰图，接入 ENG_PROF / Chrome Trace / nsys |
| **CLI（Agent 接口）** | `tc` 命令行与服务器通信，流式报文、报表文件、测试/调试结果 |
| **多 GPU** | 按 rank 编排多进程训练、逐卡监控、指标聚合与对比分析 |
| 开放 API | 完整 REST + SSE，供外部监控软件/Agent 读取 |

设计原则：

1. **低侵入**：引擎无需改代码即可被管理（环境变量 + 文件探针）；深度集成通过可选 header-only SDK。
2. **单进程服务端**：`tcd` 一个进程承载 HTTP/SSE/任务编排/监控；训练等重负载全部是受管子进程。
3. **文件即数据库**：所有持久化数据为 JSON/JSONL/CSV 明文，可直接被第三方工具消费，无嵌入式 DB 依赖。
4. **离线前端**：静态资源全部本地化（uPlot 本地 vendor），不依赖 CDN。

---

## 2. 总体架构

```
┌─────────────────────────────── Browser (HTML5 + JS) ───────────────────────────────┐
│  Dashboard │ Jobs │ Train Wizard │ Grad Oscilloscope │ Profiler │ GPUs │ Logs/CLI    │
└──────────────┬─────────────────────────────────────────────────────┬───────────────┘
               │ REST /api/v1/*                          SSE /api/v1/events (实时推送)
┌──────────────┴─────────────────────────────────────────────────────┴───────────────┐
│                              tcd  (TensorConsole Daemon)                            │
│  servic::Server (Boost.Asio io_context, 协程 accept, 每请求工作线程, ≤64 并发)        │
│ ┌──────────┬──────────┬───────────┬──────────┬───────────┬──────────┬─────────────┐ │
│ │ API 路由  │ EventBus │ JobManager│ Store    │ Probe     │ Profiler │ SysMonitor  │ │
│ │ REST/SSE │ 发布订阅  │ 进程编排   │ JSONL持久 │ 梯度采集   │ 耗时解析  │ GPU/CPU/内存 │ │
│ └──────────┴──────────┴─────┬─────┴──────────┴─────┬─────┴──────────┴─────────────┘ │
│                             │ CreateProcess/fork   │ tail/parse                     │
└─────────────────────────────┼──────────────────────┼────────────────────────────────┘
                              ▼                      ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────────────┐
   │ cmake/ninja  │  │ docker CLI   │  │ 训练子进程 (MetaModel train, rank 0..N) │
   │ 构建子进程    │  │ systemctl    │  │  ENG_DUMP_GRAD → grad CSV (每步每权重)  │
   └──────────────┘  └──────────────┘  │  ENG_PROF=1   → [prof] 耗时表 (stdout)  │
                                       │  <out>_log.csv → step 指标              │
                                       │  <out>.STOP   → 优雅停止                │
                                       └──────────────────────────────────────┘
        ▲ 可选深度集成：sdk/tc_probe.hpp（梯度/耗时实时 UDP+JSONL 上报，精确拓扑排序）
```

### 2.1 进程与线程模型

- `io_context` 单线程跑协程 accept/读请求；每个请求分发到独立工作线程（servic.cpp 内建，上限 64）。
- SSE 长连接各占一个工作线程，连接内用互斥锁串行化写；心跳 5s 一次，写失败即摘除订阅。
- 后台线程：`SysMonitor`（2s 周期）、每个运行中 Job 一个 `Watcher`（tail 指标/梯度/日志）、`EventBus` 无独立线程（发布者线程内同步分发）。
- 子进程输出经管道由 Watcher 读取，写入 `stdout.log` 并广播 `job.log` 事件。

### 2.2 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| HTTP/SSE | servic.cpp 2.0（`rt::router` + `on_stream`） | 指定基础；协程+asio，单头文件 |
| JSON | nlohmann/json（单头文件 vendor） | 事实标准，MIT |
| 时序图表 | uPlot（本地 vendor，~50KB） | 高性能 Canvas 时序，离线 |
| 示波器/火焰图 | 原生 Canvas 2D 自绘 | 定制热力/波形/火焰交互 |
| 构建 | CMake ≥ 3.18 + Ninja，g++ ≥ 13（C++20）/ MSVC 可选 | 与 TensorN/MetaModel 一致 |
| GPU 监控 | `nvidia-smi --query-gpu/--query-compute-apps` CSV | 无 NVML 链接依赖，跨驱动稳定 |
| 进程控制 | Win32 CreateProcess + JobObject / POSIX fork+exec+setpgid | 跨平台，可杀进程树 |

---

## 3. 目录结构

```
TensorConsole/
├── CMakeLists.txt
├── docs/DESIGN.md               本文档
├── third_party/nlohmann/json.hpp
├── src/                         tcd 服务端
│   ├── common.hpp/.cpp          工具/HTTP 助手/配置
│   ├── events.hpp/.cpp          EventBus + SSE 连接管理
│   ├── store.hpp/.cpp           Project/Job 持久化 + 时序读写
│   ├── process.hpp/.cpp         跨平台子进程（管道/进程树/停止）
│   ├── jobs.hpp/.cpp            JobManager：build/train/eval/test/docker/deploy/custom
│   ├── adapter.hpp/.cpp         MetaModel 适配：参数排序/CSV 解析/stdout 解析
│   ├── profiler.hpp/.cpp        [prof] 表解析 + Chrome Trace 归一化
│   ├── monitor.hpp/.cpp         GPU/CPU/内存采样
│   ├── api.cpp                  全部 REST/SSE 路由 + 静态文件
│   └── main.cpp                 入口（端口/数据目录/token 配置）
├── sdk/tc_probe.hpp / .py       引擎探针（C++ header-only / PyTorch；梯度+耗时）
├── cli/tc.cpp                   tc CLI（Agent 接口）
├── web/                         前端（index.html + css + js + vendor/uplot）
├── data/                        运行时数据（gitignore）
└── tests/smoke.py               端到端冒烟（REST 断言）
```

---

## 4. 数据模型与存储

### 4.1 实体

```jsonc
// Project —— 一个可被管理的引擎仓库
{
  "id": "metamodel",
  "name": "MetaModel",
  "root": "D:/Developments/ML/CXX-Backend/MetaModel",
  "kind": "metamodel",              // metamodel | generic
  "build": {
    "source_dir": ".", "build_dir": "build",
    "generator": "Ninja", "cuda_arch": "120",
    "targets": ["train","eval_ce","generate","gradcheck","flash_test","birkhoff_test","bench_gemm"],
    "windows_script": "build.bat"   // 存在时优先用项目自带脚本
  },
  "docker": { "dockerfile": "deploy/Dockerfile", "context": ".", "image": "metamodel:latest" },
  "created": "2026-11-17T12:00:00Z"
}

// Job —— 一次受管任务
{
  "id": "j-20261117-120000-a1b2",
  "project_id": "metamodel",
  "type": "train",                  // build|train|eval|generate|test|docker|deploy|custom
  "title": "smoke 16 steps",
  "spec": { /* 见 §8.5 全量参数 */ },
  "status": "running",              // pending|running|succeeded|failed|stopped
  "ranks": [ {"rank":0,"gpu":0,"pid":1234,"out":"checkpoints/tc_smoke/tc_smoke"} ],
  "pid": 1234,
  "created": "...", "started": "...", "ended": "...",
  "exit_code": null,
  "env": { "ENG_DUMP_GRAD": "data/jobs/<id>/grad_raw.csv", "ENG_PROF": "1" },
  "artifacts": [ {"name":"grad_raw.csv","path":"...","kind":"grad"} ]
}
```

### 4.2 磁盘布局（`data/`）

```
data/
├── projects.json                  项目注册表
├── config.json                    服务配置（端口/token/数据目录）
└── jobs/<job_id>/
    ├── job.json                   Job 记录（状态变更即写）
    ├── stdout.log                 合并 stdout/stderr（带 rank 前缀）
    ├── metrics.jsonl              {"step":12,"t":1731840000123,"rank":0,"v":{"ce":6.8,"lr":4e-5,"grad_norm":3.8,"tps":1820,"sec":1.2}}
    ├── grad_raw.csv               引擎原始 dump（step,name,grad_l2）
    ├── grad.jsonl                 {"step":12,"rank":0,"norms":[["head_norm.weight",0.31],...]}（已按示波器顺序排列）
    ├── grad_order.json            [{"name":"head_norm.weight","module":"head","depth":0},...]
    ├── profile.json               归一化耗时数据（§8.2）
    ├── report.md / report.json    CLI/面板生成的汇总报表
    └── artifacts/                 上传/下载的报表、图表、trace 文件
```

内存侧：每 Job 维护 ring buffer（指标 5000 点、梯度 300 step、日志 2000 行）用于 SSE 追新与快速查询；历史查询走 JSONL 扫描（带 from/to/limit 过滤）。

---

## 5. REST API 规范（/api/v1）

统一约定：JSON UTF-8；错误体 `{"error":"...","code":404}`；时间 ISO8601 UTC；
鉴权（可选）：`Authorization: Bearer <token>` 或 `?token=`；默认绑定 127.0.0.1，无 token。

### 5.1 系统/监控

| 方法 路径 | 说明 |
|---|---|
| GET `/health` | `{ok,version,uptime_s,jobs_running}` |
| GET `/system` | 主机名/OS/CPU 核数/内存/负载 |
| GET `/gpus` | 当前 GPU 快照数组 + 历史 ring（`?history=1`） |
| GET `/gpus/:index` | 单卡详情 + 其上的计算进程（pid→job 映射） |

### 5.2 项目

| 方法 路径 | 说明 |
|---|---|
| GET `/projects` | 列表 |
| POST `/projects` | 注册 `{name,root,kind,build?,docker?}`；自动探测 CMakeLists/build.bat/deploy |
| GET `/projects/:id` | 详情（含探测到的 targets/脚本） |
| DELETE `/projects/:id` | 注销（不删文件） |
| GET `/projects/:id/schema` | 该项目的任务参数 JSON Schema（驱动前端表单，§8.5） |

### 5.3 任务

| 方法 路径 | 说明 |
|---|---|
| GET `/jobs?status=&project=&type=&limit=` | 列表（新→旧） |
| POST `/jobs` | 提交任务（body=spec，§8.5）→ 返回 Job |
| GET `/jobs/:id` | 详情 |
| POST `/jobs/:id/stop` | 优雅停止（train 写 STOP 文件→超时 SIGINT→超时 kill 进程树） |
| DELETE `/jobs/:id` | 删除记录与数据（仅终态） |
| GET `/jobs/:id/logs?offset=&limit=` | 日志增量（行号偏移） |
| GET `/jobs/:id/metrics?keys=ce,grad_norm&from=&to=&limit=&rank=` | 时序点 |
| GET `/jobs/:id/grad/order` | 示波器参数顺序 manifest |
| GET `/jobs/:id/grad?step=&from=&to=&rank=` | 梯度步数据（缺省最新） |
| GET `/jobs/:id/profile` | 归一化耗时（summary+spans） |
| POST `/jobs/:id/profile` | 上传 Chrome Trace JSON / 指定 trace 文件路径 |
| GET `/jobs/:id/artifacts` | 产物列表 |
| GET `/jobs/:id/artifacts/:name` | 下载产物（报表/CSV/trace） |
| GET `/jobs/:id/report?format=md|json` | 生成并返回汇总报表（自动落盘 artifacts） |

### 5.4 Docker / 部署

| 方法 路径 | 说明 |
|---|---|
| GET `/docker/images` / `/docker/containers` | docker CLI 查询结果 |
| POST `/docker/build` | `{project_id,dockerfile?,context?,tag?,build_args?}` → docker 类型 Job |
| POST `/docker/run` | `{image,gpus?,mounts?,env?,command?,name?,detach?}` → Job |
| POST `/docker/stop` | `{container}` |
| POST `/deploy` | `{project_id,mode:"docker"|"systemd"|"script",...}` → deploy Job |
| GET `/deploy/services` | systemd 服务列表（Linux；Windows 返回不支持说明） |

### 5.5 事件流（SSE）

`GET /api/v1/events?topics=job,metrics,grad,gpu`（`text/event-stream`）

```
event: job.status
data: {"job_id":"j-...","status":"running","ts":...}

event: job.log
data: {"job_id":"j-...","rank":0,"line":"  step    16  ce 6.8712 ...","seq":123}

event: metrics
data: {"job_id":"j-...","step":16,"rank":0,"v":{"ce":6.87,"grad_norm":3.82,"tps":1820}}

event: grad.step
data: {"job_id":"j-...","step":16,"rank":0,"n":94}   // 大数据走 REST 拉取，事件只通知

event: gpu
data: {"gpus":[{"index":0,"util":97,"mem_used":9216,"mem_total":16311,"temp":63,"power":145}],
       "host":{"cpu_pct":12.4,"mem_pct":43,"mem_used_mb":13706,"mem_total_mb":31858,"ts":1730000000000},
       "cluster":{"count":1,"util":97,"util_max":97,"mem_used":9216,"mem_total":16311,
                  "mem_pct":56.5,"power":145,"temp_max":63}}
// cluster = 跨卡汇总：util 为各卡均值、mem/power 为合计、temp_max/util_max 为峰值；
// 采集不到的指标一律为 -1（nvidia-smi 的 [N/A]），且不参与均值/合计。
// GET /api/gpus 返回同构对象，另附 history 数组（?history=0 关闭）。

event: prof.ready
data: {"job_id":"j-..."}

event: heartbeat
data: {"ts":...}
```

---

## 6. 梯度示波器（重点 1）

### 6.1 排序规则（硬约束）

1. **模块间**：从输出到输入（梯度反传方向）。loss 侧最先，embedding 最后。
2. **模块内**：按数学表达式从左到右。线性单元 `y = Wx + b` → 先 `W` 后 `b`。
3. **串行子层**：按反传到达顺序（forward 的逆序），如 MoE 内 `down.weight` 先于 `gate_up.weight`。
4. **并行分支**（专家/多头）：同深度，按索引升序稳定排列。

MetaModel 的规范顺序（由 `adapter` 依据 `_config.json` 的 `n_decoder=S, n_experts=E` 生成）：

```
head_norm.weight                      ← 最靠近 loss
merge.res.{S-1}                       ← 最终双随机读出 R_out
for li = S-1 .. 0:
    merge.res.{li}                    (li ≥ 1)
    decoder_layers.{li}.moe.experts.{e}.down.weight      e = 0..E-1
    decoder_layers.{li}.moe.experts.{e}.gate_up.weight
    decoder_layers.{li}.moe.gate.weight / .bias
    decoder_layers.{li}.norm_moe.weight
    decoder_layers.{li}.attn.out.weight
    decoder_layers.{li}.attn.qkv.weight
    decoder_layers.{li}.norm_attn.weight
    decoder_layers.{li}.rt.w2.weight / .bias
    decoder_layers.{li}.rt.w1.weight / .bias
    decoder_layers.{li}.rt.group_gate.weight / .bias
    decoder_layers.{li}.norm_rt.weight
emb.weight                            ← 最靠近输入（tied 输出头另计 head 侧）
```

> 注：模板顺序是"反传到达顺序"的静态近似；`tc_probe.hpp` 深度集成时由引擎在 tape 上
> 做真实拓扑排序（`depth` = 距 loss 的最长路径，`expr` = 表达式内序号），覆盖模板。

### 6.2 采集通道（三级，自动降级）

| 级别 | 通道 | 侵入性 | 数据 |
|---|---|---|---|
| L0 | `ENG_DUMP_GRAD=<path>` CSV tail | 零（MetaModel 现成） | 每步每参数 `step,name,grad_l2` |
| L1 | `sdk/tc_probe.hpp`（C++）/ `sdk/tc_probe.py`（PyTorch） | 引擎/脚本加 3 行代码 | 同上 + 精确拓扑 order + rank + 可选 grad 统计(max/mean/zero 比例) |
| L2 | 引擎自定义 POST `/jobs/:id/grad` | 任意 | 任意来源（Python 桥接等） |

L0 流程：JobManager 启动 train 时注入 `ENG_DUMP_GRAD=data/jobs/<id>/grad_raw.csv`；
Watcher 增量 tail → 按 step 分组 → 按 manifest 排序 → 写 `grad.jsonl` → 广播 `grad.step`。
首个 step 到达时若尚无 manifest，用 adapter 生成（或读 SDK 上报的 order）。

### 6.3 可视化

- **瀑布热力图（主视图）**：X = 参数序号（输出→输入，模块分带标注），Y = step（向下滚动），
  颜色 = log10(grad_l2)（viridis 色标，0/NaN 灰显）。实时追加，可暂停回看。
- **单步波形（示波器模式）**：X = 参数序号，Y = grad_l2（log 轴），叠加上一 step 残影；
  step 滑杆 + 播放（1/2/5/10 step/s），直观看梯度"波前"如何从输出传向输入。
- **参数时序**：点击任一参数 → 下方 uPlot 曲线（该权重 grad_l2 随 step），可多选对比。
- **异常标注**：grad=0（死支路）、尖峰（>10× 中位数）、NaN/Inf 自动高亮并在事件流告警。
- 多 rank：rank 选择器 + 差异热力图（|rank_i − rank_0|）。

---

## 7. 全流程耗时分析（重点 2）

### 7.1 数据源

1. **ENG_PROF（零侵入）**：train 注入 `ENG_PROF=1`（可选 `ENG_PROF_NODES=1` 逐 tape 节点）。
   结束时 stdout 输出 `[prof] --- forward/backward (total X ms) ---` +
   `[prof] <tag> n=<count> <ms> ms <pct>%`。Watcher 解析为算子级 summary。
2. **Chrome Trace Event Format（标准接入）**：`POST /jobs/:id/profile` 上传
   `{"traceEvents":[{ph:"X",name,cat,pid,tid,ts,dur,args}]}`，或 SDK 直接落盘。
   支持 nsys（`nsys export --type=json`）、perfetto、自定义工具产物。
3. **tc_prof SDK**：`TC_PROF_SCOPE("name")` 宏（CPU steady_clock / CUDA event 双模式），
   线程安全缓冲，退出时写 Chrome Trace JSON；可选 UDP 实时上报。
4. **nsys 适配器（Linux）**：deploy/test Job 可包一层 `nsys profile -o <artifacts>/trace ...`，
   结束后自动 `nsys stats --report cuda_gpu_trace,cuda_api_trace --format csv` 导入。

### 7.2 归一化模型（profile.json）

```jsonc
{
  "source": "eng_prof | chrome_trace | nsys",
  "total_ms": 1234.5,
  "summary": [ {"name":"step:bwd","kind":"phase","count":16,"total_ms":800,"avg_ms":50,"pct":64.8},
               {"name":"gemm_bf16","kind":"op","count":480,"total_ms":410,"avg_ms":0.85,"pct":33.2} ],
  "spans": [ {"name":"gemm_bf16","cat":"op","tid":0,"ts_us":1200,"dur_us":850,"depth":2} ],
  "phases": {"data":..,"fwd":..,"ce":..,"bwd":..,"gradnorm":..,"opt":..}
}
```

### 7.3 可视化

- **火焰图**：Canvas 绘制，按 tid/线程分层，点击缩放、搜索高亮、耗时占比着色。
- **算子/函数表**：name / 调用次数 / 总耗时 / 平均 / 最大 / 占比，可排序、过滤 fwd/bwd。
- **阶段条**：step 内 data→fwd→ce→bwd→gradnorm→opt 占比堆叠条。
- **对比**：选择两个 Job 的 profile 做 diff（优化前后回归检查）。

---

## 8. 任务编排与重点功能

### 8.1 JobManager 与进程控制

- `Process` 抽象：Windows `CreateProcessW` + 管道 + JobObject（`KILL_ON_JOB_CLOSE` 杀进程树）；
  POSIX `fork/execvp` + `setpgid` + `kill(-pgid)`。stdout/stderr 合并读取线程。
- 停止语义（train）：写 `<out>.STOP`（引擎每步检查，保存 `_interrupt.pt` 后退出）→
  15s 未退出发 SIGINT/TerminateProcess → 再 10s 杀进程树。状态机 pending→running→(succeeded|failed|stopped)。
- 崩溃检测：exit_code≠0 → failed；stdout 尾部 200 行附入 report。
- 并发策略：同项目同类型 train 默认互斥（可 `force` 覆盖）；build 与 train 互斥（避免覆盖二进制）。

### 8.2 多 GPU（重点 4）

MetaModel 现状为单 GPU 单流，无 NCCL。TensorConsole 的多卡方案分两阶段：

- **阶段 A（v1，进程编排）**：`spec.gpus=[0,1,..]` → 每卡一个 rank 进程，
  `CUDA_VISIBLE_DEVICES=<i>`，独立 `--out <base>_rank<i>`、独立数据分片（`--data_offset`）。
  逐 rank 采集指标/梯度；面板提供 rank 对比视图与聚合（mean/min/max）。
  适用：超参扫描、数据并行实验、逐卡性能分析。
- **阶段 B（v2，真 DDP）**：引擎接入 NCCL 后，SDK 上报 allreduce 耗时/同步等待，
  面板增加通信-计算重叠timeline、rank 间梯度一致性校验（示波器 diff 视图复用）。
  API 与数据模型已预留 `rank` 维度，无需破坏性变更。

监控：`nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory` 将 GPU 进程映射到 Job/rank；
每卡 util/mem/temp/power 2s 采样入 ring，连同跨卡 cluster 汇总一起 SSE 推送。

### 8.3 CLI —— `tc`（重点 3）

C++ 实现，HTTP/1.1 客户端（Boost.Asio 同步 TCP），默认 `http://127.0.0.1:8686`
（`--server` / 环境变量 `TC_SERVER` / `~/.tensorconsole.json` 覆盖）。
所有命令支持 `--json`（Agent 机器可读）与人类可读表格双模式；退出码=任务终态。

```
tc health | system | gpu [--watch]
tc project add <name> <root> [--kind metamodel] | list | rm <id> | schema <id>
tc build <project> [--target train] [--arch 120] [--clean] [--follow]
tc train <project> [--steps N] [--out PATH] [--scratch] [--dtype 0|1]
           [--param key=value ...] [--gpus 0,1] [--prof] [--grad] [--follow] [--wait]
tc eval <project> <ckpt> <val> [--seq_len 512] [--windows 32]
tc generate <project> <ckpt> --prompt ids.bin [--max_new 64]
tc test <project> <target> [args...]        # gradcheck / flash_test / birkhoff_test
tc jobs [--status running] | job <id> | stop <id> | rm <id>
tc logs <id> [--follow] [--tail N]
tc metrics <id> [--keys ce,grad_norm] [--last N] [--csv out.csv]
tc grad <id> [--step N] [--order] [--csv out.csv]
tc prof <id> [--json out.json] [--upload trace.json]
tc report <id> [--out report.md] [--format md|json]
tc docker images|ps|build <project>|run <image> [--gpus all] [--mount ...]
tc deploy <project> [--mode docker|systemd|script]
tc events [--topics job,metrics,grad]       # 原始 SSE 流（Agent 流式获取报文）
```

Agent 工作流示例：`tc build metamodel --follow --json` → 解析编译报文 →
`tc train metamodel --steps 200 --grad --prof --wait --json` →
`tc grad j-xxx --csv g.csv` + `tc report j-xxx --out r.md` → 读取报表决策。

### 8.4 Docker / Linux 服务控制

- Docker：全部经 `docker` CLI 子进程（build/run/ps/images/stop/logs），不链接 docker SDK；
  run 参数全量映射：`--gpus`、`-v` 挂载（datasets:ro / checkpoints:rw）、`-e` 环境、
  `--network`、`--shm-size`、`--ipc=host`、restart 策略、entrypoint 覆盖。
- systemd（Linux）：`systemctl start/stop/restart/status <unit>`、`journalctl -u -f` 接入日志流；
  面板提供服务列表与启停按钮。Windows 上该模块返回明确的 unsupported。
- deploy Job：组合动作（docker build → run / 脚本执行），每步是一个子 Job，父 Job 聚合状态。

### 8.5 任务配置全量参数（重点 5）

`GET /projects/:id/schema` 返回分组 JSON Schema，前端据此动态渲染表单（含默认值/范围/枚举/帮助文本）。
MetaModel train 分组（与 engine/tools/train.cpp parse() 一一对应）：

| 组 | 参数 |
|---|---|
| 运行 | out, ckpt, resume, scratch, force, init(pytorch\|zero\|small), seed, stop_file |
| 数据 | data, val, data_offset, seq_len, batch, grad_accum, head_chunk |
| 结构 | d_model, n_heads, n_decoder, n_groups, n_experts, top_k, moe_ratio, vocab_size, max_seq_len, layer_rope, moe_rs |
| 优化 | lr, min_lr, wd, clip, ga_norm(mean\|sum), opt(adamw\|muon), mu_beta, mu_rel, mu_rel_max, mu_ns, mu_every, mu_min_dim, mu_prec, agc, warmup, total_steps |
| 损失 | aux_scale, ce_weight, aux_norm |
| 精度 | dtype(0=fp32\|1=bf16), tf32 |
| 冻结 | freeze(none\|fwd\|rev), fz_window, fz_min_steps, fz_min_rel |
| 日志/保存 | log_interval, eval_interval, eval_batches, save_interval |
| 诊断(面板注入) | ENG_DUMP_GRAD, ENG_PROF, ENG_PROF_NODES, ENG_UNBUF, ENG_TRACE |
| 多卡(面板) | gpus[], rank 数据分片策略 |

build 组：target[], cuda_arch, clean, build_type, extra_cmake_args, use_project_script。
docker 组：dockerfile, context, tag, build_args{}, gpus, mounts[], env{}, network, shm_size, ipc, restart, command, detach。
deploy 组：mode, unit_name, script_path, env{}, health_check。
generic 项目：custom command + argv + cwd + env + watch 规则（CSV 路径/列映射、stdout 正则）。

### 8.6 PyTorch 项目（kind=pytorch）

PyTorch 训练接入沿用「受管子进程 + 文件探针」模型，不引入 Python 运行时依赖：

- **提交**：`train` job 按项目 kind 分派。pytorch 参数：`python`、`script`、`args`
  （引号感知切分）、`cwd`、`env`、`out`（STOP 文件前缀）、`gpus`、
  `launcher=python|ddp`、`master_port`。`ddp` 由 tcd 直接按 gpus 每卡拉起一个
  worker（注入 `RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT`，
  与 torchrun 的子进程契约一致，但不依赖 torchrun 的 rendezvous）。
- **注入**：`TC_JOB_ID/TC_JOB_DIR/TC_RANK/TC_SERVER/TC_TOKEN/TC_STOP_FILE`、
  `CUDA_VISIBLE_DEVICES`、`PYTHONPATH=<sdk>`、`PYTHONUNBUFFERED=1`。
- **采集**：`sdk/tc_probe.py` 写 `metrics.jsonl`（src=probe，单次 os.write 保证多 rank
  追加原子性）/`grad_raw_r<rank>.csv`/`grad_order.json`/`trace.json`，与 C++ SDK 同构；
  ddp 的其他 rank 写 `trace_r<rank>.json`，finalize 时合并解析。Watcher 增加
  metrics.jsonl tail（仅转发非内部 src 的行，避免与 stdout/CSV 重复推送）。
- **梯度顺序**：优先 SDK 声明的 `grad_order.json`（默认 reversed named_parameters，
  output→input），其次 MetaModel 模板，最后按首见顺序。
- **rank**：worker 的 rank 取自 `LOCAL_RANK`；grad tail 按 worker 数量展开，
  每个 worker 独立 `grad_raw_r<rank>.csv`，停止文件按 rank 前缀分别写入。

---

## 9. 前端设计

单页应用，hash 路由，深色主题，CSS Grid 布局，无框架无构建。

| 页面 | 内容 |
|---|---|
| `#/dashboard` | GPU 卡片（util/mem/temp/power 迷你曲线）、主机负载、运行中任务、事件 ticker |
| `#/projects` | 项目卡片 + 注册表单（自动探测结果预览） |
| `#/jobs` | 任务表格（状态过滤）+ 「新建任务」向导：选项目→选类型→分组表单（schema 驱动）→提交 |
| `#/jobs/:id` | Tab：概览（loss/tps/lr/grad_norm/显存 uPlot 实时曲线 + 状态头）/ **梯度示波器** / **耗时分析** / 日志（终端样式，follow 开关）/ 产物与报表 / 配置回显 |
| `#/gpus` | 每卡历史曲线 + 计算进程表 |

组件：`api.js`（REST 封装 + SSE 订阅总线）、`charts.js`（uPlot 封装）、
`oscilloscope.js`（热力/波形/播放）、`flame.js`（火焰图）、`form.js`（schema→表单）、`app.js`（路由/视图）。
实时策略：SSE 通知 + 增量 REST 拉取（梯度大数据不塞 SSE）；断线自动重连（指数退避）。

---

## 10. 安全与可靠性

- 默认绑定 127.0.0.1；`--bind 0.0.0.0` 时强制要求 `--token`（Bearer 校验，SSE 支持 query token）。
- 静态文件与产物下载做路径规范化，禁止 `..` 逃逸；项目 root 之外的路径拒绝注册。
- 子进程命令一律 argv 数组传参（不经 shell 拼接），custom 类型显式标注风险并由用户确认。
- 请求体上限 8MB（trace 上传）；JSONL 追加写 + 定期 flush；job.json 原子替换（tmp+rename）。
- tcd 重启后扫描 data/jobs：running 态且 pid 已不存在 → 标记 failed（附重启说明）。

## 11. 测试与验收（MetaModel 冒烟）

1. `tc project add metamodel D:/Developments/ML/CXX-Backend/MetaModel --kind metamodel`
2. `tc build metamodel --target train`（已构建则秒过）
3. `tc train metamodel --scratch --steps 16 --dtype 0 --d_model 256 --n_decoder 4 --n_experts 4 --grad --prof --follow`
   （mini 数据集；注入 ENG_DUMP_GRAD/ENG_PROF）
4. 断言：metrics.jsonl 有 ce/lr/grad_norm/tps；grad.jsonl 每步 ≥90 参数且顺序符合 §6.1；
   profile.json 含 fwd/bwd 算子表；SSE 收到 job.status/metrics/grad.step/gpu。
5. `tc stop` 验证 STOP 文件优雅退出与 `_interrupt.pt`；`tc report` 生成报表。
6. 前端逐页截图核对：曲线实时刷新、示波器热力/波形/播放、火焰图缩放。

## 12. 路线图

- **v1（本次交付）**：上述全部 L0 集成 + 面板 + CLI + 多卡进程编排 + MetaModel 适配。
- **v1.1**：tc_probe SDK 接入 MetaModel（精确拓扑序、实时 UDP）、nsys 适配器、profile diff。
- **v2**：NCCL DDP 通信分析、systemd 深度管理、告警规则引擎（grad 尖峰/loss 平台/显存 OOM  webhook）、多用户与 RBAC。