# 【StudyHack】TensorConsole：把一次模型训练拆开看，逐 step 回放梯度流与 MoE 路由坍缩

> 天池「2026 Qoder 高校创造力大赛」作品帖 · StudyHack 学习科研赛道
> 发布地址：https://tianchi.aliyun.com/competition/entrance/532536/forum
> 标题填上面 H1 全文，正文从「自我介绍」开始。

## 自我介绍

我是 **Yauntyours**（GitHub / X：[@yauntyour](https://github.com/yauntyour)），个人博客「墨痕」：<https://yauntyours.cn>，写线性注意力、语言模型与工程实践的笔记。

方向是 **C++ / CUDA 与 Agent/LLM 基础设施**，习惯从底层自己造：

- **张量与引擎**：从零手搓张量加速库 [TensorN](https://github.com/yauntyour/TensorN)（C++/CUDA/OpenBLAS，零拷贝、易部署），并在其上实现 GPT 全套训练推理引擎 [CXX-GPT](https://github.com/yauntyour/CXX-GPT)、Qwen3.5-0.8B 多模态推理引擎 [CXX-Qwen](https://github.com/yauntyour/CXX-Qwen)。
- **Agent 基础设施**：[cell.cpp](https://github.com/yauntyour/cell.cpp)（轻量 AI 编码智能体，接 OpenAI / Anthropic，沙箱内读写执行代码）、[agent.cpp](https://github.com/yauntyour/agent.cpp)（Tools + Skills + 跨会话记忆）。
- **研究兴趣**：机器学习、递归自进化（RSI）、视觉-语言-动作模型（VLA）、神经网络预测控制（NNPC）、自动化工作流系统设计（AWSD）、机器人控制系统设计（RCSD）。代表作 [ELA（Exact Linear Attention）](https://arxiv.org/abs/2605.18848)：利用核函数的精确可分解性，把 O(L²) 注意力改写为线性复杂度。

<!-- TODO: 可选补充学校 / 专业 / 年级。博客未公开此项，故未填写。高校赛建议补上。 -->

**TensorConsole 就是给上面这条自研引擎链路做的训练可视化面板**——自己写引擎、自己训模型，所以格外清楚「看不见模型内部」有多难受。本次独立参赛。

## 作品是什么

TensorConsole 是一个训练过程回放面板。这次提交的是**可以直接在浏览器里玩的公开回放版**：

**<https://tensorconsole-replay-jsm88nwukl7.qoder.zone>**

里面播放的是一次**真实训练的完整录像**，不是演示数据：MetaModel（10 层 × 8 专家，d_model=512，vocab 64000，seq_len 4096，bf16），跑了 900 步、636 秒、11592 tok/s，之后被优雅停止（val_ce 8.3706，全程最优 8.2602）。**301 个权重、每一个 step 的梯度全部采集下来了。**

三个联动视图共享同一个 step 回放轴：

- **架构数据流**：把模型真实前向图画成可操纵节点图（emb → RT 门控 → 组路由 → Flash-Attn → MoE top-K → Sinkhorn 双随机混合总线 → tied head）。青绿粒子是前向数据，琥珀粒子是反向梯度。每个 MoE 节点内嵌 8 个专家格，**斜线 = 这一步没有 token 路由到它，琥珀边 = 长期饥饿**。支持拖拽平移、滚轮缩放、小地图跳转、节点 / 专家检查器。
- **MoE 路由**：层 × 专家负载矩阵、均衡度趋势（有效专家数 exp(H)、归一化熵、负载 CV、激活率，双 Y 轴）、专家覆盖表（路由频率 / 峰值负载 / 最长连续未路由 / 死亡-饥饿-间歇-活跃状态）、逐层均衡表（点击跳转架构图对应层）。
- **权重热力图**：301 权重 × 900 步，log10 通量色带 + 波形条，红色刻痕标出零梯度权重，行按层分组并显示该层步内梯度能量占比，紧凑语义别名（`L9·moe·e0·gate_up`）与完整参数名可切换。

![架构数据流 · step 900](https://tensorconsole-replay-jsm88nwukl7.qoder.zone/img/fig-arch.png)

*架构数据流 · step 900：L3 / L4 / L5 的 MoE 节点。专家格中**斜线 = 该 step 无 token 路由**，**琥珀边 = 饥饿**；L3 的 e0 与 L4 / L5 的 e3 独占 100% 梯度能量，左侧为 Sinkhorn 双随机读出行 R2–R4，青绿 / 琥珀粒子为前向数据 / 反向梯度。*

另有同一份录制的指标曲线（10 组）、完整日志（116 行）、自动报表，以及 PNG / JSON / CSV 导出。

## 为什么做

我自己训 MoE 模型时踩到一个坑：**loss 曲线看着正常，模型内部其实已经学坏了。**

专家路由会坍缩——名义上 8 个专家，训着训着只剩 1 个在接 token，其余 7 个彻底饿死。这件事在 loss 曲线上几乎看不出来，等到发现时算力已经白烧了几百步。

现有工具给的是标量曲线，回答不了「**哪个专家、在哪一步、没接到 token**」。我要的是一个能把训练过程倒回去逐 step 看的东西，于是做了 TensorConsole。

**这次录像里就抓到了这个问题**，点开就能复现：

| step | 状态 |
|---|---|
| 1–60 | 80 个专家全部在接 token，路由均衡（有效专家数 ≈ 7.7/8，归一化熵 ≈ 0.98） |
| 100 | 10 个专家开始零梯度，L8 单专家已占 76% 能量 |
| 300 | **53 / 80 个专家在这一步完全没被路由到**（106 个权重零梯度） |
| 900 | 仍有 36 / 80 个专家零梯度（72 个权重为零）；L3 / L4 / L5 梯度能量几乎 100% 集中在单个专家 |

从均衡到塌缩只用了 300 步。把滑杆从 900 拉回 1，就能看着专家格一片片亮回来。

**同一次录制，step 1 与 step 900 的层 × 专家路由矩阵对比：**

![step 1：路由均衡](https://tensorconsole-replay-jsm88nwukl7.qoder.zone/img/fig-moe-step1.png)

*step 1：10 层全部 **8/8 激活**，负载 7–20% 均匀分布，没有死专家。*

![step 900：路由坍缩](https://tensorconsole-replay-jsm88nwukl7.qoder.zone/img/fig-moe-step900.png)

*step 900：L3 / L4 / L5 / L6 / L8 塌缩到**单专家 100%**；斜线 = 死专家，琥珀边 = 饥饿，右列为该层激活数（2/8、3/8、4/8…）。*

![L4 权重热力图 · 前 120 步](https://tensorconsole-replay-jsm88nwukl7.qoder.zone/img/fig-heat.png)

*权重热力图 · L4 的 29 个权重 × 前 120 步：**step 91 起 e7 首次零梯度**，随后 e5 / e6 / e0 等行间歇断流（暗红格 + 红色刻痕）——这是坍缩的早期形态，此时 loss 还看不出异常。*

> **指标口径说明**：零梯度判定是**精确的**（该 step 无 token 路由到该专家）；负载 / 能量占比是梯度代理指标 `g^p` 归一化，真实 token 计数需要引擎侧 dispatch 遥测。面板里对这一点有明确标注，没有把代理指标当成真实负载。

## 怎么用（5 步）

1. 打开上面的链接，在弹窗里点「**开始回放**」（想自己控制就点「我自己看」）。
2. 默认停在 step 900。把 **step 滑杆往回拉到 1**，看专家格从大面积斜线变回全亮——坍缩过程倒着看一遍。
3. 切到「**MoE 路由**」，看层 × 专家矩阵和均衡度趋势，L3 / L4 / L5 的归一化熵掉到 0。
4. 切到「**权重热力图**」，301 权重 × 900 步全貌，红色刻痕就是零梯度；点任一格跳到该 step。
5. 要留证据就点「导出 PNG / JSON / CSV」；「指标曲线」「日志」「报表」tab 是同一份录制的其他侧面。

**不需要安装、不需要登录、不需要 GPU。拖动滑杆就行。**

## 怎么用 Qoder 做出来的

**整个 TensorConsole 从设计到实现都是在 Qoder 里构造的**，不是只用了补全：

- **总体设计**：`docs/DESIGN.md` 的架构、数据模型、REST/SSE 契约、梯度排序规则、三级探针采集通道（L0 零侵入 / L1 header-only SDK / L2 自定义上报）都是在 Qoder 里推演定稿的。
- **C++20 后端**：`tcd` 守护进程的 8 个模块（api / jobs / store / events / monitor / adapter / profiler / process），基于 servic.cpp（Boost.Asio 协程单头文件服务端）。跨平台子进程控制（Win32 JobObject 杀进程树 / POSIX setpgid）、file-as-database 持久化、SSE 发布订阅、优雅停止语义（STOP 文件 → SIGINT → 强杀）都在 Qoder 里实现和调试。
- **`tc` CLI 与 SDK**：Agent 可直接管道消费的 CLI（`--wait` 时退出码 = 任务退出码，报表走 stderr、任务 JSON 走 stdout），以及零依赖 header-only 探针 `tc_probe.hpp`。
- **前端**：无构建步骤的原生 JS + Canvas/uPlot SPA，三个梯度视图全部自绘，静态资源本地化、离线可用。
- **测试与发布**：`tests/smoke.py` 端到端冒烟（23 项检查）、三平台（linux-x86_64 / macos-arm64 / windows-x86_64）CI/Release 工作流。

**本次参赛版本，又在 Qoder 里做了一轮完整改造与部署：**

原来的 TensorConsole 是本地守护进程（Boost.Asio + 子进程编排 + nvidia-smi + CUDA 训练），只能在自己机器上跑，评委没法访问。改造过程：

1. **定位改造点**：Qoder 读通前端后确认所有网络请求都收敛在 `web/js/api.js` 一个对象上，可视化模块本身不直接 fetch——所以只需加一层静态数据 shim，三个视图代码可以完全不动。
2. **数据压缩**：原始 `grad.jsonl` 14.4MB（每行重复存 301 个参数名）。利用 `oscilloscope.js` 已支持的位置数组格式压到 **2.0MB**（gzip 后 0.70MB），900 步全保留。
3. **离线回放层**：新增 `demo.js` 接管 API/SSE，按原 REST 契约把录制数据喂回去；隐藏需要真实守护进程的功能（提交任务 / 构建 / Docker / 部署 / 停止），全站只读。
4. **给评委做引导**：首访弹窗用大白话解释三个视图，并把上面那张坍缩表直接放进去。
5. **主题变量接入**：设计 token 收敛到 `--site-*`，并让 Canvas 内部背景也从 CSS 变量解析，外观 / 圆角 / 间距 / 字号 / 动效五项都能实时调节。
6. **部署**：用 **Qoder Sites** 打包发布并设为公开访问。

**验证**（本地与线上分别做过，不是只看构建成功）：三视图渲染、900 步回放与滑杆联动、10 组指标图、116 行日志、报表渲染、PNG/JSON/CSV 导出、五项主题变量传导到 DOM 与画布、无控制台报错、无横向溢出、全部请求走静态路径。线上 step 300 显示 106 个零梯度权重，与独立脚本算出的 53 个死亡专家 × 2 权重完全吻合。

**使用的 Qoder 账号**：<!-- TODO: 必填。填写你的 Qoder 账号 --> 〔待填〕

## 补充

- 完整开源仓库（C++ 后端 / `tc` CLI / `tc_probe.hpp` SDK / 设计文档）：<https://github.com/yauntyour/TensorConsole>
- 公开回放版是完整工具的**只读回放子集**。耗时分析（火焰图 / Chrome Trace / nsys）需要引擎开启 `ENG_PROF`，本次录制未开启，因此回放版未展示该 tab；本地完整版支持。
- 多 GPU 逐 rank 采集与对比在本地完整版可用；本次录制为单 rank。
- 个人博客：<https://yauntyours.cn>

---

## 发布前检查清单（此节不属于正文，发布时删除）

- [ ] 填写 **Qoder 账号**（赛制硬性要求，缺失可能影响初筛）
- [ ] 决定是否补充学校 / 专业 / 年级
- [ ] 确认署名方式：当前写「本次独立参赛」，如有队友改为团队署名与分工
- [ ] 标题含赛道前缀 `【StudyHack】`
- [ ] 正文五节齐全：自我介绍 / 作品是什么 / 为什么做 / 怎么用（≤5 步）/ 怎么用 Qoder 做出来的
- [ ] 公网链接可匿名访问（已验证：`https://tensorconsole-replay-jsm88nwukl7.qoder.zone` 返回 200）
- [x] 四张配图已随站点发布并验证匿名可访问（`/img/fig-*.png`，均 200 image/png）；本地副本在 `docs/img/fig-*.png`
- [ ] 所有参赛成员已完成天池实名认证 + 阿里云大学生认证（否则取消获奖资格）
- [ ] 在 **2026-11-30 18:00** 前完成报名与提交