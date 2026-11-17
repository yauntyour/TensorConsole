#pragma once
// tc_probe.hpp — TensorConsole L1 probe SDK (header-only, zero dependencies, C++17).
//
// tcd spawns training jobs with TC_JOB_DIR / TC_RANK / TC_SERVER set; when this
// header sees TC_JOB_DIR it writes probe data into the job directory in exactly
// the formats tcd already ingests — no network, no extra threads, no engine
// changes beyond including this file:
//
//   #include "tc_probe.hpp"
//   tc::Probe probe;                              // auto-enabled under tcd
//   probe.grad(step, "decoder_layers.0.rt.w1.weight", l2);   // 梯度示波器
//   probe.metrics(step, {{"ce", ce}, {"tps", tps}});         // 自定义指标
//   { TC_PROBE_SCOPE(probe, "fwd/attention"); ... }          // 耗时 span
//   probe.flush();                                // trace.json（任务结束时也可自动）
//
// Without TC_JOB_DIR every call is a cheap no-op, so the same binary runs
// standalone unchanged.
#ifndef TC_PROBE_HPP
#define TC_PROBE_HPP

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace tc
{
    class Probe
    {
    public:
        Probe()
        {
            if (const char *d = std::getenv("TC_JOB_DIR"))
                dir_ = d;
            if (const char *r = std::getenv("TC_RANK"))
                rank_ = std::atoi(r);
            if (const char *g = std::getenv("ENG_DUMP_GRAD"))
                grad_path_ = g;
            enabled_ = !dir_.empty();
        }
        ~Probe() { flush(); }
        Probe(const Probe &) = delete;
        Probe &operator=(const Probe &) = delete;

        bool enabled() const { return enabled_; }
        const std::string &job_dir() const { return dir_; }

        // One weight's gradient L2 norm at a step (output->input ordering is
        // applied by tcd, the engine may emit in any order).
        void grad(int64_t step, const std::string &name, double l2)
        {
            if (!enabled_)
                return;
            std::lock_guard<std::mutex> lk(mx_);
            std::string path = grad_path_.empty()
                                   ? dir_ + "/grad_raw_r" + std::to_string(rank_) + ".csv"
                                   : grad_path_;
            append(path, std::to_string(step) + "," + name + "," + num(l2) + "\n");
        }

        // Arbitrary per-step metrics; merged with engine CSV/stdout series.
        void metrics(int64_t step, const std::vector<std::pair<std::string, double>> &kv)
        {
            if (!enabled_)
                return;
            std::string v = "{";
            for (size_t i = 0; i < kv.size(); ++i)
            {
                if (i)
                    v += ",";
                v += "\"" + esc(kv[i].first) + "\":" + num(kv[i].second);
            }
            v += "}";
            std::lock_guard<std::mutex> lk(mx_);
            append(dir_ + "/metrics.jsonl",
                   "{\"step\":" + std::to_string(step) + ",\"t\":" + std::to_string(now_ms()) +
                       ",\"rank\":" + std::to_string(rank_) + ",\"v\":" + v + ",\"src\":\"probe\"}\n");
        }

        // Timing spans (Chrome Trace Event format, ph=X). Nesting is allowed;
        // each thread gets its own tid so the flame graph separates threads.
        void begin(const std::string &name)
        {
            if (!enabled_)
                return;
            std::lock_guard<std::mutex> lk(mx_);
            open_.push_back({name, now_us(), tid()});
        }
        void end()
        {
            if (!enabled_)
                return;
            std::lock_guard<std::mutex> lk(mx_);
            if (open_.empty())
                return;
            auto s = open_.back();
            open_.pop_back();
            int64_t dur = now_us() - s.ts;
            events_.push_back("{\"name\":\"" + esc(s.name) + "\",\"cat\":\"probe\",\"ph\":\"X\"," +
                              "\"ts\":" + std::to_string(s.ts) + ",\"dur\":" + std::to_string(dur) +
                              ",\"pid\":0,\"tid\":" + std::to_string(s.tid) + "}");
        }

        // Write trace.json; tcd ingests it when the job finalizes (or upload
        // live with: tc profile <job> --upload <dir>/trace.json).
        void flush()
        {
            if (!enabled_)
                return;
            std::lock_guard<std::mutex> lk(mx_);
            if (events_.empty())
                return;
            std::string out = "{\"displayTimeUnit\":\"us\",\"traceEvents\":[";
            for (size_t i = 0; i < events_.size(); ++i)
            {
                if (i)
                    out += ",";
                out += events_[i];
            }
            out += "]}";
            std::ofstream f(dir_ + "/trace.json", std::ios::binary | std::ios::trunc);
            f << out;
        }

        struct Scope
        {
            Probe &p;
            Scope(Probe &pr, const std::string &name) : p(pr) { p.begin(name); }
            ~Scope() { p.end(); }
            Scope(const Scope &) = delete;
            Scope &operator=(const Scope &) = delete;
        };

    private:
        struct Open
        {
            std::string name;
            int64_t ts;
            uint64_t tid;
        };

        static int64_t now_us()
        {
            return std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }
        static int64_t now_ms()
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                .count();
        }
        static uint64_t tid()
        {
            return (uint64_t)std::hash<std::thread::id>{}(std::this_thread::get_id());
        }
        static std::string num(double v)
        {
            char buf[48];
            std::snprintf(buf, sizeof buf, "%.9g", v);
            return buf;
        }
        static std::string esc(const std::string &s)
        {
            std::string out;
            for (char c : s)
            {
                if (c == '"' || c == '\\')
                    out += '\\', out += c;
                else if ((unsigned char)c < 0x20)
                    out += ' ';
                else
                    out += c;
            }
            return out;
        }
        static void append(const std::string &path, const std::string &data)
        {
            std::ofstream f(path, std::ios::binary | std::ios::app);
            f << data;
        }

        bool enabled_ = false;
        std::string dir_, grad_path_;
        int rank_ = 0;
        std::mutex mx_;
        std::vector<Open> open_;
        std::vector<std::string> events_;
    };
}

#define TC_PROBE_SCOPE(probe, name) ::tc::Probe::Scope _tc_probe_scope_(probe, name)

#endif // TC_PROBE_HPP