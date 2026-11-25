#include "jobs.hpp"

#include "adapter.hpp"
#include "profiler.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <set>

namespace tc
{
    JobManager::JobManager(Config &cfg, Store &store, EventBus &bus)
        : cfg_(cfg), store_(store), bus_(bus)
    {
    }

    JobManager::~JobManager()
    {
        std::vector<Live *> all;
        {
            std::lock_guard<std::mutex> lk(mx_);
            for (auto &kv : live_)
                all.push_back(kv.second.get());
        }
        for (auto *L : all)
        {
            L->stop_req = true;
            L->stop_deadline = 0;
            for (auto &r : L->ranks)
                if (r.proc)
                    r.proc->terminate(false);
            if (L->phase_proc)
                L->phase_proc->terminate(false);
        }
        for (auto *L : all)
            if (L->watcher.joinable())
                L->watcher.join();
        std::lock_guard<std::mutex> lk(mx_);
        live_.clear();
    }

    JobManager::Live *JobManager::live(const std::string &id)
    {
        std::lock_guard<std::mutex> lk(mx_);
        auto it = live_.find(id);
        return it == live_.end() ? nullptr : it->second.get();
    }

    std::map<std::string, std::string> JobManager::kv_env(const std::string &s)
    {
        std::map<std::string, std::string> out;
        for (auto &kv : split(s, ';'))
        {
            auto t = trim(kv);
            auto eq = t.find('=');
            if (eq != std::string::npos && eq > 0)
                out[trim(t.substr(0, eq))] = trim(t.substr(eq + 1));
        }
        return out;
    }

    // CLI `-p gpus=0` is typed as a JSON number; accept any scalar as a string.
    static std::string jstr(const J &j, const char *key, const std::string &def = "")
    {
        auto it = j.find(key);
        if (it == j.end() || it->is_null())
            return def;
        return it->is_string() ? it->get<std::string>() : it->dump();
    }

    // Locate sdk/tc_probe.py so Python train jobs can `from tc_probe import Probe`
    // without touching the script (injected via PYTHONPATH).
    static fs::path probe_sdk_dir(const Config &cfg)
    {
        if (const char *e = std::getenv("TCD_SDK_DIR"))
        {
            fs::path p = e;
            if (fs::exists(p / "tc_probe.py"))
                return p;
        }
        std::vector<fs::path> cands = {
            fs::path(cfg.web_dir).parent_path() / "sdk",
            fs::current_path() / "sdk",
            fs::path(cfg.data_dir).parent_path() / "sdk",
        };
        for (auto &c : cands)
        {
            std::error_code ec;
            if (fs::exists(c / "tc_probe.py", ec))
                return c;
        }
        return {};
    }

    void JobManager::ensure_train_slot(const std::string &project_id, const J &params)
    {
        if (params.value("force", false))
            return;
        std::lock_guard<std::mutex> lk(mx_);
        for (auto &kv : live_)
            if (kv.second->rec.value("project_id", "") == project_id &&
                kv.second->rec.value("type", "") == "train" && !kv.second->done)
                throw std::runtime_error("another train job is running on this project (use force)");
    }

    // ------------------------------------------------------------- submit

    static std::vector<std::string> option_argv(const J &schema, const std::string &type,
                                                const J &params, std::vector<std::string> positional)
    {
        std::vector<std::string> argv = std::move(positional);
        if (!schema.contains(type))
            return argv;
        for (auto &g : schema[type]["groups"])
            for (auto &fld : g["fields"])
            {
                std::string key = fld.at("key").get<std::string>();
                if (!params.contains(key))
                    continue;
                const J &v = params[key];
                std::string type_s = fld.value("type", "string");
                std::string s = v.is_string() ? v.get<std::string>() : v.dump();
                if (type_s == "flag")
                {
                    if (v.is_boolean() ? v.get<bool>() : s == "true")
                        argv.push_back("--" + key);
                }
                else if (!s.empty())
                {
                    argv.push_back("--" + key);
                    argv.push_back(s);
                }
            }
        return argv;
    }

    std::vector<ProcSpec> JobManager::docker_phases(const J &proj, const J &params)
    {
        fs::path root = proj.value("root", ".");
        std::string action = params.value("action", "run");
        ProcSpec sp;
        sp.exe = "docker";
        sp.cwd = root.string();
        if (action == "build")
        {
            std::string df = params.value("dockerfile", proj.value("docker", J::object()).value("dockerfile", "deploy/Dockerfile"));
            std::string tag = params.value("tag", proj.value("docker", J::object()).value("image", "tensorconsole:latest"));
            sp.argv = {"build", "-f", df, "-t", tag};
            for (auto &kv : kv_env(params.value("build_args", "")))
            {
                sp.argv.push_back("--build-arg");
                sp.argv.push_back(kv.first + "=" + kv.second);
            }
            sp.argv.push_back(params.value("context", "."));
        }
        else if (action == "stop")
        {
            sp.argv = {"stop", params.value("container", "")};
        }
        else
        {
            sp.argv = {"run"};
            if (params.value("detach", true))
                sp.argv.push_back("-d");
            std::string gpus = jstr(params, "gpus", "all");
            if (!gpus.empty())
            {
                sp.argv.push_back("--gpus");
                sp.argv.push_back(gpus);
            }
            for (auto &m : split(params.value("mounts", ""), ';'))
                if (!trim(m).empty())
                {
                    sp.argv.push_back("-v");
                    sp.argv.push_back(trim(m));
                }
            for (auto &kv : kv_env(params.value("env", "")))
            {
                sp.argv.push_back("-e");
                sp.argv.push_back(kv.first + "=" + kv.second);
            }
            if (!params.value("network", "").empty())
            {
                sp.argv.push_back("--network");
                sp.argv.push_back(params["network"].get<std::string>());
            }
            if (!params.value("shm_size", "").empty())
            {
                sp.argv.push_back("--shm-size");
                sp.argv.push_back(params["shm_size"].get<std::string>());
            }
            if (params.value("ipc_host", false))
                sp.argv.push_back("--ipc=host");
            if (!params.value("restart", "").empty())
            {
                sp.argv.push_back("--restart");
                sp.argv.push_back(params["restart"].get<std::string>());
            }
            if (!params.value("name", "").empty())
            {
                sp.argv.push_back("--name");
                sp.argv.push_back(params["name"].get<std::string>());
            }
            sp.argv.push_back(params.value("image", proj.value("docker", J::object()).value("image", "")));
            for (auto &a : split(params.value("command", ""), ' '))
                if (!trim(a).empty())
                    sp.argv.push_back(trim(a));
        }
        return {sp};
    }

    std::vector<ProcSpec> JobManager::deploy_phases(const J &proj, const J &params)
    {
        std::string mode = params.value("mode", "docker");
        if (mode == "docker")
        {
            J p = params;
            if (p.value("image", "").empty())
            {
                p["action"] = "build";
                auto build = docker_phases(proj, p);
                p["action"] = "run";
                p["image"] = proj.value("docker", J::object()).value("image", "");
                auto run = docker_phases(proj, p);
                build.insert(build.end(), run.begin(), run.end());
                return build;
            }
            p["action"] = "run";
            return docker_phases(proj, p);
        }
        ProcSpec sp;
        if (mode == "systemd")
        {
#ifdef _WIN32
            throw std::runtime_error("systemd deploy is Linux-only");
#else
            sp.exe = "systemctl";
            sp.argv = {params.value("action", "status"), params.value("unit", "")};
#endif
        }
        else
        {
            std::string script = params.value("script", "");
            if (script.empty())
                throw std::runtime_error("deploy script mode requires 'script'");
#ifdef _WIN32
            sp.exe = "cmd.exe";
            sp.argv = {"/c", script};
#else
            sp.exe = "bash";
            sp.argv = {script};
#endif
            sp.cwd = proj.value("root", ".");
        }
        return {sp};
    }

    ProcSpec JobManager::build_spec(const J &proj, const J &params, std::vector<ProcSpec> &phases)
    {
        fs::path root = proj.value("root", ".");
        J b = proj.value("build", J::object());
        std::string build_dir = b.value("build_dir", "build");
        std::string arch = params.value("cuda_arch", b.value("cuda_arch", "120"));
        std::string btype = params.value("build_type", "Release");
        std::vector<std::string> targets;
        for (auto &t : split(params.value("targets", "train"), ','))
            if (!trim(t).empty())
                targets.push_back(trim(t));
        bool clean = params.value("clean", false);
        bool use_script = params.value("use_script", true);
        phases.clear();

#ifdef _WIN32
        if (use_script && fs::exists(root / "build.bat"))
        {
            if (clean)
                phases.push_back(ProcSpec{"cmd.exe", {"/c", "build.bat", "clean"}, root.string(), {}});
            std::vector<std::string> argv = {"/c", "build.bat"};
            for (auto &t : targets)
                argv.push_back(t);
            if (!arch.empty())
            {
                argv.push_back("-arch");
                argv.push_back(arch);
            }
            phases.push_back(ProcSpec{"cmd.exe", argv, root.string(), {}});
            return {};
        }
#endif
        if (clean)
            phases.push_back(ProcSpec{"cmake", {"-E", "rm", "-rf", build_dir}, root.string(), {}});
        std::vector<std::string> cfg = {"-S", ".", "-B", build_dir, "-G", "Ninja",
                                        "-DCMAKE_BUILD_TYPE=" + btype};
        if (!arch.empty())
            cfg.push_back("-DCMAKE_CUDA_ARCHITECTURES=" + arch);
        for (auto &a : split(params.value("extra", ""), ' '))
            if (!trim(a).empty())
                cfg.push_back(trim(a));
        phases.push_back(ProcSpec{"cmake", cfg, root.string(), {}});
        std::vector<std::string> bargv = {"--build", build_dir, "-j"};
        for (auto &t : targets)
        {
            bargv.push_back("--target");
            bargv.push_back(t);
        }
        phases.push_back(ProcSpec{"cmake", bargv, root.string(), {}});
        return {};
    }

    J JobManager::submit(const J &body)
    {
        std::string project_id = body.value("project_id", "");
        std::string type = body.value("type", "");
        J proj = store_.get_project(project_id);
        if (proj.is_null())
            throw std::runtime_error("unknown project: " + project_id);
        fs::path root = proj.at("root").get<std::string>();
        const std::string kind = proj.value("kind", "generic");
        J params = body.value("params", J::object());
        std::string id = new_id("j");
        fs::path jd = store_.job_dir(id);
        ensure_dir(jd);

        J rec = {{"id", id},
                 {"project_id", project_id},
                 {"type", type},
                 {"title", body.value("title", type + " " + id)},
                 {"root", root.string()},
                 {"spec", {{"params", params}}},
                 {"status", "running"},
                 {"created", iso_utc()},
                 {"started", iso_utc()},
                 {"ended", nullptr},
                 {"exit_code", nullptr},
                 {"ranks", J::array()},
                 {"artifacts", J::array()}};

        auto L = std::make_unique<Live>();
        L->rec = rec;
        auto emit = [this, Lp = L.get()](int rank, const std::string &line) {
            on_line(Lp, rank, line);
        };

        if (type == "train" && (kind == "pytorch" || kind == "python"))
        {
            ensure_train_slot(project_id, params);
            J pt = proj.value("train", J::object());
            fs::path cwd = jstr(params, "cwd", root.string());
            if (cwd.is_relative())
                cwd = root / cwd;
            std::string script = jstr(params, "script", pt.value("script", "train.py"));
            fs::path spath = script;
            if (spath.is_relative())
                spath = cwd / spath;
            if (script.empty() || !fs::exists(spath))
                throw std::runtime_error("python script not found: " + spath.string() +
                                         " — set params.script");
            std::string python = jstr(params, "python", pt.value("python", "python"));
            if (python.empty())
                python = "python";
            std::string out_base = jstr(params, "out", "checkpoints/tc_" + id);
            std::vector<int> gpus;
            for (auto &g : split(jstr(params, "gpus", "0"), ','))
                if (!trim(g).empty())
                    gpus.push_back(std::stoi(trim(g)));
            if (gpus.empty())
                gpus.push_back(0);
            std::string gpu_list;
            for (int g : gpus)
                gpu_list += (gpu_list.empty() ? "" : ",") + std::to_string(g);
            // ddp: tcd spawns one worker per GPU with torchrun-compatible env
            // (RANK/WORLD_SIZE/LOCAL_RANK/MASTER_*), no torchrun process needed.
            std::string launcher = jstr(params, "launcher", "python");
            bool ddp = launcher == "ddp" || launcher == "torchrun";
            int nranks = ddp ? (int)gpus.size() : 1;
            std::vector<std::string> args = split_quoted(jstr(params, "args"));

            std::map<std::string, std::string> env = kv_env(jstr(params, "env"));
#ifdef _WIN32
            // Windows wheels ship without libuv and TCPStore defaults to libuv;
            // env:// init_process_group aborts unless USE_LIBUV=0.
            if (!env.count("USE_LIBUV") && !std::getenv("USE_LIBUV"))
                env["USE_LIBUV"] = "0";
#endif
            env["PYTHONUNBUFFERED"] = "1";
            env["CUDA_VISIBLE_DEVICES"] = gpu_list;
            env["MASTER_ADDR"] = "127.0.0.1";
            env["MASTER_PORT"] = jstr(params, "master_port", "29500");
            env["WORLD_SIZE"] = std::to_string(nranks);
            // L1 probe SDK (sdk/tc_probe.py) reads these inside the training script.
            env["TC_JOB_ID"] = id;
            env["TC_JOB_DIR"] = fs::absolute(jd).string();
            env["TC_SERVER"] = "http://" + cfg_.bind + ":" + std::to_string(cfg_.port);
            if (!cfg_.token.empty())
                env["TC_TOKEN"] = cfg_.token;
            if (fs::path sdk = probe_sdk_dir(cfg_); !sdk.empty())
            {
                std::string pp = fs::absolute(sdk).string();
                env["TC_SDK_DIR"] = pp;
#ifdef _WIN32
                const char *sep = ";";
#else
                const char *sep = ":";
#endif
                if (const char *old = std::getenv("PYTHONPATH"); old && *old)
                    pp += std::string(sep) + old;
                env["PYTHONPATH"] = pp;
            }
            J cmd = J::array();
            cmd.push_back(python);
            cmd.push_back(script);
            for (auto &a : args)
                cmd.push_back(a);
            rec["cmd"] = cmd;

            for (int i = 0; i < nranks; ++i)
            {
                ProcSpec sp;
                sp.exe = python;
                sp.cwd = cwd.string();
                sp.argv.push_back(script);
                for (auto &a : args)
                    sp.argv.push_back(a);
                sp.env = env;
                sp.env["RANK"] = std::to_string(i);
                sp.env["LOCAL_RANK"] = std::to_string(i);
                sp.env["TC_RANK"] = std::to_string(i);
                Rank rk;
                rk.rank = i;
                rk.gpu = gpus[i % (int)gpus.size()];
                rk.out = nranks > 1 ? out_base + "_rank" + std::to_string(i) : out_base;
                sp.env["TC_STOP_FILE"] = fs::absolute(root / (rk.out + ".STOP")).string();
                rk.proc = std::make_unique<Process>();
                int rank = i;
                if (!rk.proc->start(sp, [emit, rank](const std::string &l) { emit(rank, l); }))
                {
                    for (auto &prev : L->ranks)
                        prev.proc->terminate(false);
                    throw std::runtime_error("failed to spawn train rank " + std::to_string(i) +
                                             " (" + python + "; check params.python / PATH)");
                }
                rec["ranks"].push_back({{"rank", i}, {"gpu", rk.gpu},
                                        {"pid", rk.proc->pid()}, {"out", rk.out}});
                if (i == 0)
                    rec["env"] = sp.env;
                L->ranks.push_back(std::move(rk));
            }
            L->tails.resize(nranks); // one grad file per worker (probe LOCAL_RANK)
        }
        else if (type == "train")
        {
            if (kind != "metamodel")
                throw std::runtime_error("train type requires a metamodel or pytorch project (use custom otherwise)");
            ensure_train_slot(project_id, params);
            fs::path exe = root / "build" / exe_name("train");
            if (!fs::exists(exe))
                throw std::runtime_error("train binary not found: " + exe.string() + " — run a build job first");
            if (!params.contains("data"))
            {
                fs::path mini = root / "datasets/cache/mix_bil3b_mini_train.bin";
                if (fs::exists(mini))
                    params["data"] = "datasets/cache/mix_bil3b_mini_train.bin";
                else
                    throw std::runtime_error("params.data is required");
            }
            if (!params.contains("val"))
            {
                fs::path mini = root / "datasets/cache/mix_bil3b_mini_val.bin";
                if (fs::exists(mini))
                    params["val"] = "datasets/cache/mix_bil3b_mini_val.bin";
            }
            std::string out_base = params.value("out", "checkpoints/tc_" + id);
            std::vector<int> gpus;
            for (auto &g : split(jstr(params, "gpus", "0"), ','))
                if (!trim(g).empty())
                    gpus.push_back(std::stoi(trim(g)));
            if (gpus.empty())
                gpus.push_back(0);
            bool grad = params.value("grad", true);
            bool prof = params.value("prof", false);
            J clean_params = params;
            clean_params.erase("out");
            int nranks = (int)gpus.size();
            int64_t shard_tokens = 0;
            if (nranks > 1 && params.value("data_shard", false))
            {
                std::error_code ec;
                auto bytes = fs::file_size(root / params["data"].get<std::string>(), ec);
                if (!ec)
                    shard_tokens = (int64_t)(bytes / 4) / nranks;
            }
            for (int i = 0; i < nranks; ++i)
            {
                Rank rk;
                rk.rank = i;
                rk.gpu = gpus[i];
                rk.out = nranks > 1 ? out_base + "_rank" + std::to_string(i) : out_base;
                J rp = clean_params;
                if (shard_tokens > 0 && !params.contains("data_offset"))
                    rp["data_offset"] = shard_tokens * i;
                ProcSpec sp;
                sp.exe = exe.string();
                sp.argv = adapter::train_argv(rp, rk.out);
                sp.cwd = root.string();
                sp.env["ENG_UNBUF"] = "1";
                sp.env["CUDA_VISIBLE_DEVICES"] = std::to_string(rk.gpu);
                // L1 probe SDK (sdk/tc_probe.hpp) picks these up inside the engine.
                sp.env["TC_JOB_ID"] = id;
                sp.env["TC_JOB_DIR"] = fs::absolute(jd).string();
                sp.env["TC_RANK"] = std::to_string(i);
                sp.env["TC_SERVER"] = "http://" + cfg_.bind + ":" + std::to_string(cfg_.port);
                if (!cfg_.token.empty())
                    sp.env["TC_TOKEN"] = cfg_.token;
                if (grad)
                    sp.env["ENG_DUMP_GRAD"] = fs::absolute(jd / ("grad_raw_r" + std::to_string(i) + ".csv")).string();
                if (prof)
                    sp.env["ENG_PROF"] = "1";
                if (params.value("prof_nodes", false))
                    sp.env["ENG_PROF_NODES"] = "1";
                if (params.value("trace", false))
                    sp.env["ENG_TRACE"] = "1";
                rk.proc = std::make_unique<Process>();
                int rank = i;
                if (!rk.proc->start(sp, [emit, rank](const std::string &l) { emit(rank, l); }))
                {
                    for (auto &prev : L->ranks)
                        prev.proc->terminate(false);
                    throw std::runtime_error("failed to spawn train rank " + std::to_string(i));
                }
                rec["ranks"].push_back({{"rank", i}, {"gpu", rk.gpu}, {"pid", rk.proc->pid()}, {"out", rk.out}});
                if (i == 0)
                    rec["env"] = sp.env;
                L->ranks.push_back(std::move(rk));
                L->tails.emplace_back();
            }
        }
        else if (type == "build")
        {
            build_spec(proj, params, L->phases);
        }
        else if (type == "eval" || type == "generate" || type == "test")
        {
            auto schema = adapter::schema_for(kind);
            std::string target = type;
            std::vector<std::string> positional;
            if (type == "test")
            {
                target = params.value("target", "gradcheck");
                for (auto &a : split(params.value("args", ""), ' '))
                    if (!trim(a).empty())
                        positional.push_back(trim(a));
            }
            else if (type == "eval")
            {
                if (kind == "metamodel")
                    target = "eval_ce";
                std::string val = params.value("val", "");
                if (val.empty() && fs::exists(root / "datasets/cache/mix_bil3b_mini_val.bin"))
                    val = "datasets/cache/mix_bil3b_mini_val.bin";
                positional = {params.value("ckpt", "")};
                if (!val.empty())
                    positional.push_back(val);
            }
            else
            {
                positional = {params.value("ckpt", "")};
            }
            if (kind == "pytorch" || kind == "python")
                throw std::runtime_error(
                    type + " is not managed for pytorch projects; submit a custom job " +
                    "(params.command=<python>, params.argv=\"<script> ...\") or add the action to your train script");
            fs::path exe = root / "build" / exe_name(target);
            if (!fs::exists(exe))
                throw std::runtime_error("binary not found: " + exe.string());
            ProcSpec sp;
            sp.exe = exe.string();
            sp.argv = option_argv(schema, type, params, positional);
            sp.cwd = root.string();
            sp.env["ENG_UNBUF"] = "1";
            L->phases.push_back(std::move(sp));
        }
        else if (type == "docker")
        {
            L->phases = docker_phases(proj, params);
        }
        else if (type == "deploy")
        {
            L->phases = deploy_phases(proj, params);
        }
        else if (type == "custom")
        {
            ProcSpec sp;
            sp.exe = params.value("command", "");
            if (sp.exe.empty())
                throw std::runtime_error("custom job requires params.command");
            for (auto &a : split(params.value("argv", ""), ' '))
                if (!trim(a).empty())
                    sp.argv.push_back(trim(a));
            sp.cwd = params.value("cwd", root.string());
            sp.env = kv_env(params.value("env", ""));
            L->phases.push_back(std::move(sp));
        }
        else
        {
            throw std::runtime_error("unknown job type: " + type);
        }

        store_.create_job(rec);
        {
            std::lock_guard<std::mutex> lk(mx_);
            live_[id] = std::move(L);
        }
        start_watcher(live_[id].get());
        bus_.publish("job.status", J{{"job_id", id}, {"status", "running"}, {"type", type}});
        return rec;
    }

    void JobManager::start_watcher(Live *L)
    {
        bool train = !L->ranks.empty();
        L->watcher = std::thread([this, L, train] {
            if (train)
                watch_train(L);
            else
                watch_phases(L);
        });
    }

    // ------------------------------------------------------------- stdout ingestion

    void JobManager::on_line(Live *L, int rank, const std::string &line)
    {
        std::string id = L->rec.at("id").get<std::string>();
        bool multi = L->ranks.size() > 1;
        store_.append_log(id, multi ? "[r" + std::to_string(rank) + "] " + line : line);
        bus_.publish("job.log", J{{"job_id", id}, {"rank", rank}, {"line", line}});
        if (L->rec.value("type", "") != "train")
            return;
        if (auto s = adapter::parse_step_line(line))
        {
            int64_t step = (*s)["step"].get<int64_t>();
            J v = {{"tps", (*s)["tps"]}, {"fm", (*s)["fm"]}, {"nodes", (*s)["nodes"]},
                   {"trainable", (*s)["trainable"]}, {"params_total", (*s)["params_total"]}};
            {
                std::lock_guard<std::mutex> lk(L->mx);
                L->last_stdout_step = step;
            }
            store_.append_jsonl(id, "metrics.jsonl",
                                J{{"step", step}, {"t", now_ms()}, {"rank", rank}, {"v", v}, {"src", "stdout"}});
            bus_.publish("metrics", J{{"job_id", id}, {"step", step}, {"rank", rank}, {"v", v}});
        }
        else if (auto p = adapter::parse_pool_line(line))
        {
            int64_t step;
            {
                std::lock_guard<std::mutex> lk(L->mx);
                step = L->last_stdout_step;
            }
            store_.append_jsonl(id, "metrics.jsonl",
                                J{{"step", step}, {"t", now_ms()}, {"rank", rank}, {"v", *p}, {"src", "pool"}});
        }
        else if (auto pr = adapter::parse_prof_line(line))
        {
            std::lock_guard<std::mutex> lk(L->mx);
            if (pr->contains("prof_header"))
            {
                L->prof_phase = (*pr)["prof_header"].get<std::string>();
                L->prof_totals[L->prof_phase] = (*pr)["total_ms"];
            }
            else if (pr->contains("prof_entry"))
            {
                J e = *pr;
                e["phase"] = L->prof_phase;
                L->prof_entries.push_back(std::move(e));
            }
        }
        else if (line.find("[val]") != std::string::npos || line.find("[best]") != std::string::npos ||
                 line.find("[save]") != std::string::npos || line.find("[unfreeze]") != std::string::npos)
        {
            bus_.publish("job.event", J{{"job_id", id}, {"rank", rank}, {"line", line}});
        }
    }

    // ------------------------------------------------------------- file tailing

    static std::string read_tail(const fs::path &p, std::streamoff &off, std::string &partial)
    {
        std::ifstream f(p, std::ios::binary);
        if (!f)
            return "";
        f.seekg(0, std::ios::end);
        auto size = (std::streamoff)f.tellg();
        if (size <= off)
        {
            if (size < off)
                off = 0; // truncated (scratch restart)
            else
                return "";
        }
        f.seekg(off);
        std::string chunk((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        off += (std::streamoff)chunk.size();
        partial += chunk;
        std::string complete;
        size_t pos;
        while ((pos = partial.find('\n')) != std::string::npos)
        {
            complete += partial.substr(0, pos + 1);
            partial.erase(0, pos + 1);
        }
        return complete;
    }

    void JobManager::tail_csv(Live *L, size_t r)
    {
        std::string id = L->rec.at("id").get<std::string>();
        fs::path root = L->rec.value("root", ".");
        fs::path p = root / (L->ranks[r].out + "_log.csv");
        auto &t = L->tails[r];
        std::string data = read_tail(p, t.csv_off, t.csv_part);
        for (auto &line : split(data, '\n'))
        {
            auto v = adapter::parse_log_csv(line);
            if (!v)
                continue;
            int64_t step = (*v)["step"].get<int64_t>();
            store_.append_jsonl(id, "metrics.jsonl",
                                J{{"step", step}, {"t", now_ms()}, {"rank", (int)r}, {"v", *v}, {"src", "csv"}});
            bus_.publish("metrics", J{{"job_id", id}, {"step", step}, {"rank", (int)r}, {"v", *v}});
        }
    }

    void JobManager::ensure_order(Live *L, const std::vector<std::string> &names)
    {
        if (L->order_set)
            return;
        std::string id = L->rec.at("id").get<std::string>();
        fs::path root = L->rec.value("root", ".");
        // An SDK-declared order (tc_probe.py watch(model)) wins over templates.
        J tmpl;
        auto order_path = store_.job_dir(id) / "grad_order.json";
        if (fs::exists(order_path))
        {
            auto j = J::parse(read_file(order_path), nullptr, false);
            if (j.is_array() && !j.empty())
                tmpl = j;
        }
        if (tmpl.is_null())
        {
            J cfg = J::object();
            auto cfg_path = root / (L->ranks[0].out + "_config.json");
            if (fs::exists(cfg_path))
            {
                auto j = J::parse(read_file(cfg_path), nullptr, false);
                if (j.is_object())
                    cfg = j;
            }
            bool metamodel = cfg.contains("n_decoder");
            for (auto &nm : names)
                if (starts_with(nm, "decoder_layers."))
                    metamodel = true;
            if (!metamodel)
            {
                // Arbitrary framework (PyTorch etc.): keep first-seen order.
                J arr = J::array();
                std::set<std::string> seen;
                for (auto &nm : names)
                {
                    if (!seen.insert(nm).second)
                        continue;
                    auto dot = nm.find('.');
                    arr.push_back({{"name", nm},
                                   {"module", nm.substr(0, dot)},
                                   {"depth", (int)arr.size()}});
                }
                L->order = arr;
                L->order_set = true;
                write_file_atomic(order_path, L->order.dump(2));
                return;
            }
            if (!cfg.contains("n_decoder"))
            {
                int max_layer = -1, max_expert = -1;
                for (auto &nm : names)
                {
                    if (starts_with(nm, "decoder_layers."))
                    {
                        auto rest = nm.substr(15);
                        auto dot = rest.find('.');
                        if (dot != std::string::npos)
                            max_layer = std::max(max_layer, std::atoi(rest.substr(0, dot).c_str()));
                        auto ep = nm.find(".moe.experts.");
                        if (ep != std::string::npos)
                        {
                            auto rest2 = nm.substr(ep + 13);
                            auto dot2 = rest2.find('.');
                            if (dot2 != std::string::npos)
                                max_expert = std::max(max_expert, std::atoi(rest2.substr(0, dot2).c_str()));
                        }
                    }
                }
                cfg["n_decoder"] = max_layer + 1;
                cfg["n_experts"] = max_expert + 1;
            }
            tmpl = adapter::grad_order_template(cfg);
        }
        L->order = adapter::order_observed(tmpl, names);
        L->order_set = true;
        write_file_atomic(order_path, L->order.dump(2));
    }

    void JobManager::flush_grad(Live *L, size_t r, int64_t step)
    {
        auto &t = L->tails[r];
        if (step < 0 || t.pend.empty())
            return;
        std::string id = L->rec.at("id").get<std::string>();
        std::vector<std::string> names;
        for (auto &pr : t.pend)
            names.push_back(pr.first);
        ensure_order(L, names);
        std::map<std::string, int> idx;
        for (size_t i = 0; i < L->order.size(); ++i)
            idx[L->order[i].at("name").get<std::string>()] = (int)i;
        auto rows = t.pend;
        std::sort(rows.begin(), rows.end(), [&](auto &a, auto &b) {
            int ia = idx.count(a.first) ? idx[a.first] : 1 << 20;
            int ib = idx.count(b.first) ? idx[b.first] : 1 << 20;
            return ia < ib;
        });
        J norms = J::array();
        for (auto &pr : rows)
            norms.push_back(J::array({pr.first, pr.second}));
        store_.append_jsonl(id, "grad.jsonl",
                            J{{"step", step}, {"t", now_ms()}, {"rank", (int)r}, {"norms", norms}});
        bus_.publish("grad.step", J{{"job_id", id}, {"step", step}, {"rank", (int)r}, {"n", norms.size()}});
        t.pend.clear();
    }

    void JobManager::tail_grad(Live *L, size_t r)
    {
        std::string id = L->rec.at("id").get<std::string>();
        fs::path p = store_.job_dir(id) / ("grad_raw_r" + std::to_string(r) + ".csv");
        auto &t = L->tails[r];
        std::string data = read_tail(p, t.grad_off, t.grad_part);
        for (auto &line : split(data, '\n'))
        {
            auto c = split(trim(line), ',');
            if (c.size() != 3)
                continue;
            try
            {
                int64_t step = std::stoll(c[0]);
                double l2 = std::stod(c[2]);
                if (step != t.cur_step)
                {
                    flush_grad(L, r, t.cur_step);
                    t.cur_step = step;
                }
                t.pend.emplace_back(c[1], l2);
            }
            catch (...)
            {
            }
        }
    }

    // ------------------------------------------------------------- watchers

    // Rows written by probes (sdk/tc_probe.*) land in metrics.jsonl directly;
    // relay them over SSE. Rows written by this watcher were already broadcast
    // at ingestion time and are skipped.
    void JobManager::tail_metrics(Live *L)
    {
        std::string id = L->rec.at("id").get<std::string>();
        fs::path p = store_.job_dir(id) / "metrics.jsonl";
        std::string data = read_tail(p, L->met_off, L->met_part);
        for (auto &line : split(data, '\n'))
        {
            auto t = trim(line);
            if (t.empty())
                continue;
            auto row = J::parse(t, nullptr, false);
            if (!row.is_object())
                continue;
            auto src = row.value("src", "");
            if (src == "stdout" || src == "csv" || src == "pool")
                continue;
            bus_.publish("metrics", J{{"job_id", id},
                                      {"step", row.value("step", (int64_t)-1)},
                                      {"rank", row.value("rank", (int64_t)0)},
                                      {"v", row.value("v", J::object())}});
        }
    }

    void JobManager::watch_train(Live *L)
    {
        std::string id = L->rec.at("id").get<std::string>();
        while (!L->done)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            bool all_exited = true;
            for (size_t r = 0; r < L->ranks.size(); ++r)
            {
                tail_csv(L, r);
                if (L->ranks[r].proc && L->ranks[r].proc->running())
                    all_exited = false;
            }
            for (size_t r = 0; r < L->tails.size(); ++r)
                tail_grad(L, r);
            tail_metrics(L);
            if (L->stop_req && now_ms() > L->stop_deadline)
                for (auto &rk : L->ranks)
                    if (rk.proc && rk.proc->running())
                        rk.proc->terminate(false);
            if (all_exited)
            {
                for (size_t r = 0; r < L->ranks.size(); ++r)
                    tail_csv(L, r);
                for (size_t r = 0; r < L->tails.size(); ++r)
                {
                    tail_grad(L, r);
                    flush_grad(L, r, L->tails[r].cur_step);
                    L->tails[r].cur_step = -1;
                }
                tail_metrics(L);
                int code = 0;
                for (auto &rk : L->ranks)
                    if (rk.proc)
                        code = std::max(code, rk.proc->exit_code().value_or(-1));
                std::string status = L->stop_req ? "stopped" : (code == 0 ? "succeeded" : "failed");
                finalize(L, status, code);
            }
        }
    }

    void JobManager::watch_phases(Live *L)
    {
        std::string id = L->rec.at("id").get<std::string>();
        for (L->phase_idx = 0; L->phase_idx < L->phases.size(); ++L->phase_idx)
        {
            if (L->stop_req)
            {
                finalize(L, "stopped", -1);
                return;
            }
            bus_.publish("job.event", J{{"job_id", id},
                                        {"line", "[phase " + std::to_string(L->phase_idx + 1) + "/" +
                                                     std::to_string(L->phases.size()) + "] " +
                                                     L->phases[L->phase_idx].exe + " " +
                                                     join(L->phases[L->phase_idx].argv, " ")}});
            auto p = std::make_unique<Process>();
            int phase = (int)L->phase_idx;
            if (!p->start(L->phases[L->phase_idx],
                          [this, L, phase](const std::string &l) { on_line(L, phase, l); }))
            {
                store_.append_log(id, "[error] failed to spawn: " + L->phases[L->phase_idx].exe);
                finalize(L, "failed", -1);
                return;
            }
            L->phase_proc = std::move(p);
            std::optional<int> code;
            while (!(code = L->phase_proc->wait(300)))
            {
                if (L->stop_req)
                {
                    L->phase_proc->terminate(false);
                    L->phase_proc->wait(3000);
                    finalize(L, "stopped", -1);
                    return;
                }
            }
            L->phase_proc.reset();
            if (code.value_or(-1) != 0)
            {
                finalize(L, "failed", code.value_or(-1));
                return;
            }
        }
        finalize(L, "succeeded", 0);
    }

    void JobManager::finalize(Live *L, const std::string &status, int exit_code)
    {
        if (L->done)
            return;
        std::string id = L->rec.at("id").get<std::string>();
        J profile;
        {
            std::lock_guard<std::mutex> lk(L->mx);
            if (!L->prof_entries.empty())
                profile = profiler::from_eng_prof(L->prof_entries, L->prof_totals);
        }
        if (profile.is_null())
        {
            // L1 SDK span traces: trace.json plus per-rank trace_r<i>.json (DDP).
            J events = J::array();
            auto merge = [&](const fs::path &p) {
                auto j = J::parse(read_file(p), nullptr, false);
                if (j.is_object() && j.contains("traceEvents") && j["traceEvents"].is_array())
                    for (auto &e : j["traceEvents"])
                        events.push_back(e);
                else if (j.is_array())
                    for (auto &e : j)
                        events.push_back(e);
            };
            fs::path tp = store_.job_dir(id) / "trace.json";
            if (fs::exists(tp))
                merge(tp);
            for (auto &f : list_dir(store_.job_dir(id)))
            {
                auto nm = f.filename().string();
                if (starts_with(nm, "trace_r") && ends_with(nm, ".json"))
                    merge(f);
            }
            if (!events.empty())
                profile = profiler::from_chrome_trace(J{{"traceEvents", events}});
        }
        if (!profile.is_null())
        {
            write_file_atomic(store_.job_dir(id) / "profile.json", profile.dump());
            bus_.publish("prof.ready", J{{"job_id", id}});
        }
        J patch = {{"status", status}, {"exit_code", exit_code}, {"ended", iso_utc()}};
        store_.update_job(id, patch);
        for (auto &kv : patch.items())
            L->rec[kv.key()] = kv.value();
        L->done = true;
        bus_.publish("job.status", J{{"job_id", id}, {"status", status}, {"exit_code", exit_code}});
    }

    // ------------------------------------------------------------- control / queries

    J JobManager::stop(const std::string &id)
    {
        Live *L = live(id);
        if (!L || L->done)
            throw std::runtime_error("job is not running: " + id);
        L->stop_req = true;
        if (!L->ranks.empty())
        {
            fs::path root = L->rec.value("root", ".");
            for (auto &rk : L->ranks)
            {
                fs::path stop = root / (rk.out + ".STOP");
                ensure_dir(stop.parent_path());
                write_file_atomic(stop, "stop requested by TensorConsole " + iso_utc() + "\n");
            }
            L->stop_deadline = now_ms() + 15000;
        }
        else
        {
            L->stop_deadline = 0;
        }
        return store_.get_job(id);
    }

    J JobManager::job_of_pid(int64_t pid)
    {
        std::lock_guard<std::mutex> lk(mx_);
        for (auto &kv : live_)
        {
            Live *L = kv.second.get();
            for (auto &rk : L->ranks)
                if (rk.proc && rk.proc->pid() == pid)
                    return J{{"job_id", L->rec.value("id", "")}, {"rank", rk.rank},
                             {"type", L->rec.value("type", "")}, {"title", L->rec.value("title", "")}};
        }
        return J();
    }

    void JobManager::recover()
    {
        for (auto &j : store_.list_jobs("running"))
            store_.update_job(j.at("id").get<std::string>(),
                              J{{"status", "failed"}, {"ended", iso_utc()},
                                {"note", "tcd restarted; child process state unknown"}});
        for (auto &j : store_.list_jobs("pending"))
            store_.update_job(j.at("id").get<std::string>(), J{{"status", "failed"}, {"ended", iso_utc()}});
    }

    std::vector<J> JobManager::metrics(const std::string &id, const std::string &keys,
                                       int64_t from, int64_t to, int64_t rank, size_t limit)
    {
        std::set<std::string> want;
        for (auto &k : split(keys, ','))
            if (!trim(k).empty())
                want.insert(trim(k));
        std::map<std::pair<int64_t, int64_t>, J> merged;
        for (auto &row : store_.read_jsonl(id, "metrics.jsonl"))
        {
            int64_t step = row.value("step", (int64_t)-1);
            int64_t r = row.value("rank", (int64_t)0);
            if (step < 0 || (rank >= 0 && r != rank))
                continue;
            if (from >= 0 && step < from)
                continue;
            if (to >= 0 && step > to)
                continue;
            auto &m = merged[{step, r}];
            if (m.is_null())
                m = J{{"step", step}, {"rank", r}, {"t", row.value("t", (int64_t)0)}, {"v", J::object()}};
            J v = row.value("v", J::object());
            for (auto &kv : v.items())
                if (want.empty() || want.count(kv.key()))
                    m["v"][kv.key()] = kv.value();
        }
        std::vector<J> out;
        for (auto &kv : merged)
            out.push_back(kv.second);
        if (limit > 0 && out.size() > limit)
            out.erase(out.begin(), out.end() - limit);
        return out;
    }

    J JobManager::grad_order(const std::string &id)
    {
        auto p = store_.job_dir(id) / "grad_order.json";
        if (fs::exists(p))
        {
            auto j = J::parse(read_file(p), nullptr, false);
            if (j.is_array())
                return j;
        }
        Live *L = live(id);
        if (L && L->order_set)
            return L->order;
        return J::array();
    }

    std::vector<J> JobManager::grad_steps(const std::string &id, int64_t step, int64_t from,
                                          int64_t to, int64_t rank, size_t limit)
    {
        std::vector<J> out;
        for (auto &row : store_.read_jsonl(id, "grad.jsonl"))
        {
            int64_t s = row.value("step", (int64_t)-1);
            int64_t r = row.value("rank", (int64_t)0);
            if (step >= 0 && s != step)
                continue;
            if (from >= 0 && s < from)
                continue;
            if (to >= 0 && s > to)
                continue;
            if (rank >= 0 && r != rank)
                continue;
            out.push_back(row);
        }
        if (limit > 0 && out.size() > limit)
            out.erase(out.begin(), out.end() - limit);
        return out;
    }

    J JobManager::profile(const std::string &id)
    {
        auto p = store_.job_dir(id) / "profile.json";
        if (fs::exists(p))
        {
            auto j = J::parse(read_file(p), nullptr, false);
            if (j.is_object())
                return j;
        }
        Live *L = live(id);
        if (L)
        {
            std::lock_guard<std::mutex> lk(L->mx);
            if (!L->prof_entries.empty())
                return profiler::from_eng_prof(L->prof_entries, L->prof_totals);
        }
        return J::object();
    }

    void JobManager::upload_profile(const std::string &id, const J &trace)
    {
        J prof = profiler::from_chrome_trace(trace);
        write_file_atomic(store_.job_dir(id) / "profile.json", prof.dump());
        bus_.publish("prof.ready", J{{"job_id", id}});
    }

    J JobManager::logs(const std::string &id, size_t offset, size_t limit)
    {
        auto lines = split(read_file(store_.job_dir(id) / "stdout.log"), '\n');
        if (!lines.empty() && lines.back().empty())
            lines.pop_back();
        size_t total = lines.size();
        if (offset > total)
            offset = total;
        size_t end = limit > 0 ? std::min(total, offset + limit) : total;
        J arr = J::array();
        for (size_t i = offset; i < end; ++i)
            arr.push_back(J{{"seq", i}, {"line", lines[i]}});
        return J{{"lines", arr}, {"next_offset", end}, {"total", total}};
    }

    J JobManager::report(const std::string &id, const std::string &format)
    {
        J job = store_.get_job(id);
        if (job.is_null())
            throw std::runtime_error("unknown job: " + id);
        auto mets = metrics(id, "", -1, -1, -1, 0);
        auto order = grad_order(id);
        auto grads = grad_steps(id, -1, -1, -1, -1, 1);
        J last_grad = grads.empty() ? J::object() : grads.back();
        J prof = profile(id);
        auto log_lines = split(read_file(store_.job_dir(id) / "stdout.log"), '\n');
        std::string tail;
        size_t start = log_lines.size() > 30 ? log_lines.size() - 30 : 0;
        for (size_t i = start; i < log_lines.size(); ++i)
            tail += log_lines[i] + "\n";
        J rep = profiler::make_report(job, mets, order, last_grad, prof, tail);
        write_file_atomic(store_.job_dir(id) / "report.md", rep["md"].get<std::string>());
        write_file_atomic(store_.job_dir(id) / "report.json", rep["json"].dump(2));
        if (format == "json")
            return rep["json"];
        return J{{"md", rep["md"]}};
    }
}