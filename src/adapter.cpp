#include "adapter.hpp"

#include <cstdio>
#include <set>

namespace tc::adapter
{
    static J f(const char *key, const char *label, const char *type, J def = J(),
               J opts = J(), const char *help = "")
    {
        J j = {{"key", key}, {"label", label}, {"type", type}};
        if (!def.is_null())
            j["def"] = def;
        if (!opts.is_null())
            j["opts"] = opts;
        if (help && help[0])
            j["help"] = help;
        return j;
    }

    static J grp(const char *name, std::vector<J> fields)
    {
        return {{"name", name}, {"fields", fields}};
    }

    static J metamodel_train_groups()
    {
        return J::array({
            grp("运行", {
                f("out", "输出前缀", "string", J(), J(), "checkpoints/<run>/<run>，生成 .pt/_best/_interrupt/_log.csv"),
                f("ckpt", "指定断点", "string", J(), J(), "显式加载的 .pt 路径"),
                f("resume", "续训断点", "string", J(), J(), "从指定断点续训"),
                f("scratch", "从零训练", "flag", false, J(), "不加载断点，截断日志"),
                f("force", "强制覆盖", "flag", false),
                f("init", "初始化方式", "enum", "pytorch", J::array({"pytorch", "zero", "small"})),
                f("steps", "训练步数", "int", 2000),
                f("seed", "随机种子", "int"),
                f("stop_file", "停止文件", "string", J(), J(), "面板停止时自动写 <out>.STOP"),
            }),
            grp("数据", {
                f("data", "训练集", "string", J(), J(), "int32 token 流 .bin"),
                f("val", "验证集", "string"),
                f("data_offset", "数据偏移(token)", "int"),
                f("seq_len", "序列长度", "int", 4096),
                f("batch", "batch", "int", 1),
                f("grad_accum", "梯度累积", "int", 2),
                f("head_chunk", "输出头分块", "int"),
            }),
            grp("结构", {
                f("d_model", "d_model", "int", 512),
                f("n_heads", "注意力头数", "int", 8),
                f("n_decoder", "decoder 层数", "int", 10),
                f("n_groups", "路由组 G", "int", 16),
                f("n_experts", "专家数 E", "int", 16),
                f("top_k", "top-k", "int", 2),
                f("moe_ratio", "MoE 隐维倍率", "float", 4.0),
                f("vocab_size", "词表大小", "int", 248077),
                f("max_seq_len", "最大序列", "int", 4096),
                f("layer_rope", "layer_rope 覆盖", "int"),
                f("moe_rs", "moe_rs 覆盖", "int"),
            }),
            grp("优化", {
                f("lr", "学习率", "float", 4e-5),
                f("min_lr", "最低学习率", "float"),
                f("wd", "权重衰减", "float"),
                f("clip", "梯度裁剪", "float"),
                f("ga_norm", "累积归一", "enum", "mean", J::array({"mean", "sum"})),
                f("opt", "优化器", "enum", "adamw", J::array({"adamw", "muon"})),
                f("mu_beta", "Muon beta", "float"),
                f("mu_rel", "Muon rel", "float"),
                f("mu_rel_max", "Muon rel_max", "float"),
                f("mu_ns", "Newton-Schulz 迭代", "int"),
                f("mu_every", "Muon 周期", "int"),
                f("mu_min_dim", "Muon 最小维度", "int"),
                f("mu_prec", "Muon 精度", "enum", J(), J::array({"fp32", "tf32"})),
                f("lr_ctrl", "在线 LR 控制", "enum", "off",
                  J::array({"off", "report", "drift", "drift+loss"}),
                  "按漂移/梯度预算在线调 lr（off|report|drift|drift+loss）"),
                f("lr_drift_budget", "漂移预算/步", "float", J(), J(),
                  "d ln‖W‖ 每步预算；空=引擎按 ln2/total_steps 推"),
                f("lr_rate_budget", "梯度增长预算/步", "float", J(), J(),
                  "d ln|g| 每步预算；空=引擎按 ln10/total_steps 推"),
                f("lr_ctrl_window", "LR 控制窗口", "int", J(), J(), "默认 256 步"),
                f("lr_ctrl_patience", "平台期窗口数", "int", J(), J(), "默认 8，仅 drift+loss"),
                f("lr_ctrl_min", "LR λ 下限", "float", J(), J(), "默认 0.05"),
                f("rnpe", "尺度保持", "enum", "off", J::array({"off", "emb", "all"}),
                  "钉住 tied emb 行范数（all 再加 head_norm RMS）"),
                f("rnpe_every", "尺度保持周期", "int", J(), J(), "默认 32 步"),
                f("agc", "AGC 系数", "float"),
                f("warmup", "warmup 步数", "int"),
                f("total_steps", "调度总步数", "int"),
            }),
            grp("损失", {
                f("aux_scale", "aux 权重", "float"),
                f("ce_weight", "CE 权重", "float"),
                f("aux_norm", "aux 归一", "string"),
            }),
            grp("精度", {
                f("dtype", "张量精度", "enum", "1", J::array({"0", "1"}), "0=fp32, 1=bf16 张量核"),
                f("tf32", "TF32", "enum", J(), J::array({"0", "1"})),
            }),
            grp("冻结", {
                f("freeze", "冻结策略", "enum", "none", J::array({"none", "fwd", "rev"})),
                f("fz_window", "冻结窗口", "int"),
                f("fz_min_steps", "冻结最少步数", "int"),
                f("fz_min_rel", "冻结最小改善", "float"),
            }),
            grp("日志/保存", {
                f("log_interval", "日志间隔", "int", 10),
                f("eval_interval", "验证间隔", "int"),
                f("eval_batches", "验证批数", "int"),
                f("save_interval", "保存间隔", "int"),
            }),
            grp("诊断（面板注入）", {
                f("grad", "梯度示波器", "flag", true, J(), "注入 ENG_DUMP_GRAD，逐 step 逐权重采集"),
                f("prof", "耗时分析", "flag", false, J(), "注入 ENG_PROF=1，结束时输出算子耗时表"),
                f("prof_nodes", "逐节点反向计时", "flag", false, J(), "ENG_PROF_NODES=1"),
                f("trace", "引擎 trace", "flag", false, J(), "ENG_TRACE=1"),
            }),
            grp("多 GPU", {
                f("gpus", "GPU 列表", "string", "0", J(), "逗号分隔，如 0,1；每卡一个 rank 进程"),
                f("data_shard", "数据分片", "flag", false, J(), "按 rank 自动偏移 data_offset"),
            }),
        });
    }

    static J pytorch_train_groups()
    {
        return J::array({
            grp("运行", {
                f("python", "Python 解释器", "string", "python", J(),
                  "如 python / python3 / .venv/Scripts/python.exe"),
                f("script", "训练脚本", "string", "train.py", J(), "相对 cwd 的路径"),
                f("args", "脚本参数", "string", J(), J(),
                  "空格分隔，支持引号：--steps 100 --data \"my data.bin\""),
                f("cwd", "工作目录", "string", J(), J(), "默认项目根目录"),
                f("out", "输出前缀", "string", J(), J(),
                  "面板停止时写 <out>.STOP；脚本用 probe.should_stop() 检测"),
                f("env", "环境变量", "string", J(), J(), "K=V;K2=V2"),
                f("force", "允许并发", "flag", false, J(), "同项目已有 train 运行时强制提交"),
            }),
            grp("多 GPU", {
                f("gpus", "GPU 列表", "string", "0", J(),
                  "逗号分隔；python=单进程可见全部，ddp=每卡一个 worker"),
                f("launcher", "启动器", "enum", "python", J::array({"python", "ddp"}),
                  "ddp=按 gpus 每卡一个进程，注入 torchrun 兼容环境变量"),
                f("master_port", "DDP master 端口", "int", 29500),
            }),
        });
    }

    J schema_for(const std::string &kind)
    {
        J s;
        if (kind == "pytorch" || kind == "python")
            s["train"] = {{"groups", pytorch_train_groups()}};
        if (kind == "metamodel")
        {
            s["train"] = {{"groups", metamodel_train_groups()}};
            s["eval"] = {{"positional", J::array({"ckpt", "val"})},
                         {"groups", J::array({grp("评测", {
                             f("seq_len", "序列长度", "int", 512),
                             f("windows", "窗口数", "int", 32),
                             f("start", "起始窗口", "int"),
                             f("chunk", "分块", "int"),
                             f("dtype", "精度", "enum", "0", J::array({"0", "1"})),
                             f("csv", "输出 CSV", "string"),
                         })})}};
            s["generate"] = {{"positional", J::array({"ckpt"})},
                             {"groups", J::array({grp("生成", {
                                 f("prompt", "prompt ids.bin", "string"),
                                 f("max_new", "新 token 数", "int", 64),
                                 f("temp", "温度", "float", 0.8),
                                 f("top_k", "top-k", "int", 40),
                                 f("dtype", "精度", "enum", "0", J::array({"0", "1"})),
                             })})}};
            s["test"] = {{"groups", J::array({grp("测试", {
                f("target", "测试目标", "enum", "gradcheck",
                  J::array({"gradcheck", "flash_test", "birkhoff_test", "fwd_test", "bench_gemm"})),
                f("args", "位置/选项参数", "string", J(), J(), "如 gradcheck: <ckpt> <ids> --tol 1e-2"),
            })})}};
        }
        s["build"] = {{"groups", J::array({grp("CMake 构建", {
            f("targets", "目标", "string", "train", J(), "逗号分隔；空=全部"),
            f("cuda_arch", "CUDA 架构", "string", "120", J(), "RTX 50 系=120, H100=90"),
            f("build_type", "构建类型", "enum", "Release", J::array({"Release", "Debug", "RelWithDebInfo"})),
            f("clean", "清理重建", "flag", false),
            f("use_script", "优先项目脚本", "flag", true, J(), "Windows 下存在 build.bat 时使用"),
            f("extra", "附加 CMake 参数", "string"),
        })})}};
        s["docker"] = {{"groups", J::array({grp("Docker", {
            f("action", "动作", "enum", "run", J::array({"build", "run", "stop", "ps", "images"})),
            f("image", "镜像", "string"),
            f("dockerfile", "Dockerfile", "string", "deploy/Dockerfile"),
            f("context", "上下文", "string", "."),
            f("tag", "构建 tag", "string"),
            f("build_args", "构建参数", "string", J(), J(), "K=V;K2=V2"),
            f("gpus", "--gpus", "string", "all"),
            f("mounts", "挂载", "string", J(), J(), "src:dst[:ro];..."),
            f("env", "环境变量", "string", J(), J(), "K=V;K2=V2"),
            f("network", "网络", "string"),
            f("shm_size", "shm 大小", "string", "8g"),
            f("ipc_host", "--ipc=host", "flag", false),
            f("restart", "restart 策略", "string"),
            f("name", "容器名", "string"),
            f("command", "容器命令", "string"),
            f("detach", "后台运行", "flag", true),
            f("container", "目标容器(stop)", "string"),
        })})}};
        s["deploy"] = {{"groups", J::array({grp("部署", {
            f("mode", "模式", "enum", "docker", J::array({"docker", "systemd", "script"})),
            f("unit", "systemd 单元", "string"),
            f("action", "systemd 动作", "enum", "status", J::array({"start", "stop", "restart", "status"})),
            f("script", "脚本路径", "string"),
            f("image", "docker 镜像", "string"),
            f("gpus", "--gpus", "string", "all"),
            f("mounts", "挂载", "string"),
            f("env", "环境变量", "string"),
            f("command", "命令", "string"),
        })})}};
        s["custom"] = {{"groups", J::array({grp("自定义命令", {
            f("command", "可执行文件", "string"),
            f("argv", "参数", "string", J(), J(), "空格分隔"),
            f("cwd", "工作目录", "string"),
            f("env", "环境变量", "string", J(), J(), "K=V;K2=V2"),
        })})}};
        return s;
    }

    // ------------------------------------------------------------- grad order

    J grad_order_template(const J &cfg)
    {
        const int S = std::max(1, cfg.value("n_decoder", 10));
        const int E = std::max(1, cfg.value("n_experts", 16));
        J out = J::array();
        int depth = 0;
        auto push = [&](const std::string &name, const std::string &module) {
            out.push_back({{"name", name}, {"module", module}, {"depth", depth}});
        };
        push("head_norm.weight", "head");
        ++depth;
        push("merge.res." + std::to_string(S - 1), "merge.out");
        ++depth;
        for (int li = S - 1; li >= 0; --li)
        {
            const std::string p = "decoder_layers." + std::to_string(li) + ".";
            const std::string L = "L" + std::to_string(li);
            if (li >= 1 && li != S - 1)
                push("merge.res." + std::to_string(li), "merge");
            for (int e = 0; e < E; ++e)
                push(p + "moe.experts." + std::to_string(e) + ".down.weight", L + ".moe");
            for (int e = 0; e < E; ++e)
                push(p + "moe.experts." + std::to_string(e) + ".gate_up.weight", L + ".moe");
            push(p + "moe.gate.weight", L + ".moe");
            push(p + "moe.gate.bias", L + ".moe");
            push(p + "norm_moe.weight", L + ".moe");
            push(p + "attn.out.weight", L + ".attn");
            push(p + "attn.qkv.weight", L + ".attn");
            push(p + "norm_attn.weight", L + ".attn");
            push(p + "rt.w2.weight", L + ".rt");
            push(p + "rt.w2.bias", L + ".rt");
            push(p + "rt.w1.weight", L + ".rt");
            push(p + "rt.w1.bias", L + ".rt");
            push(p + "rt.group_gate.weight", L + ".rt");
            push(p + "rt.group_gate.bias", L + ".rt");
            push(p + "norm_rt.weight", L + ".rt");
            ++depth;
        }
        push("emb.weight", "emb");
        return out;
    }

    J order_observed(const J &tmpl, const std::vector<std::string> &names)
    {
        std::set<std::string> have(names.begin(), names.end());
        std::set<std::string> used;
        J out = J::array();
        for (auto &e : tmpl)
        {
            auto nm = e.value("name", "");
            if (have.count(nm) && !used.count(nm))
            {
                out.push_back(e);
                used.insert(nm);
            }
        }
        std::vector<std::string> rest;
        for (auto &nm : names)
            if (!used.count(nm))
                rest.push_back(nm);
        std::sort(rest.begin(), rest.end());
        for (auto &nm : rest)
            out.push_back({{"name", nm}, {"module", "other"}, {"depth", 999}});
        return out;
    }

    // ------------------------------------------------------------- parsers

    std::optional<J> parse_step_line(const std::string &line)
    {
        long long step = 0, trim = 0;
        double ce = 0, aux = 0, fm = 0, lr = 0, g = 0, tps = 0;
        size_t tr = 0, tot = 0, nodes = 0;
        int n = std::sscanf(line.c_str(),
                            " step %lld ce %lf aux %lf fm %lf lr %lf |g| %lf train %zu/%zu %lf tok/s nodes %zu",
                            &step, &ce, &aux, &fm, &lr, &g, &tr, &tot, &tps, &nodes);
        if (n != 10)
            return std::nullopt;
        (void)trim;
        return J{{"step", step}, {"ce", ce}, {"aux", aux}, {"fm", fm}, {"lr", lr},
                 {"grad_norm", g}, {"trainable", tr}, {"params_total", tot},
                 {"tps", tps}, {"nodes", nodes}};
    }

    std::optional<J> parse_pool_line(const std::string &line)
    {
        double used = 0, reserved = 0, peak = 0, free = 0;
        long long trim = 0;
        int n = std::sscanf(line.c_str(), " [pool] used %lf reserved %lf peak %lf trim %lld driver_free %lf MiB",
                            &used, &reserved, &peak, &trim, &free);
        if (n != 5)
            return std::nullopt;
        return J{{"pool_used", used}, {"pool_reserved", reserved}, {"pool_peak", peak},
                 {"pool_trim", trim}, {"driver_free", free}};
    }

    std::optional<J> parse_prof_line(const std::string &line)
    {
        if (!starts_with(trim(line), "[prof]"))
            return std::nullopt;
        char phase[32] = {0};
        double tot = 0;
        if (std::sscanf(line.c_str(), "[prof] --- %31s (total %lf ms) ---", phase, &tot) == 2)
            return J{{"prof_header", phase}, {"total_ms", tot}};
        char tag[128] = {0};
        size_t count = 0;
        double ms = 0, pct = 0;
        if (std::sscanf(line.c_str(), "[prof] %127s n=%zu %lf ms %lf%%", tag, &count, &ms, &pct) == 4)
            return J{{"prof_entry", tag}, {"count", count}, {"ms", ms}, {"pct", pct}};
        return std::nullopt;
    }

    std::optional<J> parse_log_csv(const std::string &line)
    {
        auto parts = split(trim(line), ',');
        // 7 列是旧格式；当前引擎在末尾追加 lrc_lambda,drift,grate,emb_rms（共 11 列）。
        if (parts.size() < 7)
            return std::nullopt;
        try
        {
            J j{{"step", std::stoll(parts[0])},
                {"ce", std::stod(parts[1])},
                {"aux", std::stod(parts[2])},
                {"lr", std::stod(parts[3])},
                {"grad_norm", std::stod(parts[4])},
                {"unfrozen", std::stoi(parts[5])},
                {"sec", std::stod(parts[6])}};
            if (parts.size() >= 11)
            {
                j["lrc_lambda"] = std::stod(parts[7]);
                j["drift"] = std::stod(parts[8]);
                j["grate"] = std::stod(parts[9]);
                j["emb_rms"] = std::stod(parts[10]);
            }
            return j;
        }
        catch (...)
        {
            return std::nullopt; // header or malformed
        }
    }

    // ------------------------------------------------------------- argv

    static std::string value_str(const J &v)
    {
        if (v.is_string())
            return v.get<std::string>();
        return v.dump();
    }

    std::vector<std::string> train_argv(const J &params, const std::string &out)
    {
        std::vector<std::string> argv;
        auto schema = schema_for("metamodel")["train"]["groups"];
        std::set<std::string> special = {"grad", "prof", "prof_nodes", "trace", "gpus", "data_shard"};
        for (auto &g : schema)
            for (auto &fld : g["fields"])
            {
                std::string key = fld.at("key").get<std::string>();
                if (special.count(key) || !params.contains(key))
                    continue;
                const J &v = params[key];
                std::string type = fld.value("type", "string");
                if (type == "flag")
                {
                    if (v.is_boolean() ? v.get<bool>() : value_str(v) == "true")
                        argv.push_back("--" + key);
                }
                else
                {
                    std::string s = value_str(v);
                    if (s.empty())
                        continue;
                    argv.push_back("--" + key);
                    argv.push_back(s);
                }
            }
        if (!out.empty())
        {
            argv.push_back("--out");
            argv.push_back(out);
        }
        return argv;
    }
}