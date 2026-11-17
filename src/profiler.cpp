#include "profiler.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace tc::profiler
{
    J from_eng_prof(const std::vector<J> &entries, const J &phase_totals)
    {
        J summary = J::array();
        J phases = J::object();
        double total = 0;
        for (auto &kv : phase_totals.items())
            total += kv.value().get<double>();
        for (auto &e : entries)
        {
            std::string name = e.value("prof_entry", "");
            double ms = e.value("ms", 0.0);
            size_t count = e.value("count", (size_t)0);
            J row = {{"name", name},
                     {"kind", starts_with(name, "step:") ? "phase" : "op"},
                     {"phase", e.value("phase", "")},
                     {"count", count},
                     {"total_ms", ms},
                     {"avg_ms", count ? ms / (double)count : 0.0},
                     {"pct", total > 0 ? 100.0 * ms / total : 0.0}};
            summary.push_back(row);
            if (starts_with(name, "step:"))
                phases[name.substr(5)] = ms;
        }
        std::sort(summary.begin(), summary.end(),
                  [](const J &a, const J &b) { return a["total_ms"] > b["total_ms"]; });
        return J{{"source", "eng_prof"}, {"total_ms", total}, {"summary", summary},
                 {"phases", phases}, {"spans", J::array()}};
    }

    J from_chrome_trace(const J &trace)
    {
        const J &events = trace.contains("traceEvents") ? trace["traceEvents"] : trace;
        struct Span
        {
            std::string name, cat;
            int pid = 0, tid = 0, depth = 0;
            double ts = 0, dur = 0;
        };
        std::vector<Span> spans;
        for (auto &e : events)
        {
            if (e.value("ph", "") != "X")
                continue;
            Span s;
            s.name = e.value("name", "?");
            s.cat = e.value("cat", "");
            s.pid = e.value("pid", 0);
            s.tid = e.value("tid", 0);
            s.ts = e.value("ts", 0.0);
            s.dur = e.value("dur", 0.0);
            spans.push_back(std::move(s));
        }
        std::sort(spans.begin(), spans.end(), [](const Span &a, const Span &b) {
            if (a.pid != b.pid) return a.pid < b.pid;
            if (a.tid != b.tid) return a.tid < b.tid;
            if (a.ts != b.ts) return a.ts < b.ts;
            return a.dur > b.dur;
        });
        // depth via per-thread stack of end times
        std::map<std::pair<int, int>, std::vector<double>> stacks;
        double min_ts = 1e300, max_end = -1e300;
        for (auto &s : spans)
        {
            auto &st = stacks[{s.pid, s.tid}];
            while (!st.empty() && st.back() <= s.ts + 1e-9)
                st.pop_back();
            s.depth = (int)st.size();
            st.push_back(s.ts + s.dur);
            min_ts = std::min(min_ts, s.ts);
            max_end = std::max(max_end, s.ts + s.dur);
        }
        double wall_us = spans.empty() ? 0 : max_end - min_ts;
        std::map<std::string, J> agg;
        J span_arr = J::array();
        for (auto &s : spans)
        {
            span_arr.push_back({{"name", s.name}, {"cat", s.cat}, {"pid", s.pid}, {"tid", s.tid},
                                {"ts_us", s.ts - (spans.empty() ? 0 : min_ts)},
                                {"dur_us", s.dur}, {"depth", s.depth}});
            auto &a = agg[s.name];
            if (a.is_null())
                a = J{{"name", s.name}, {"kind", "op"}, {"count", 0}, {"total_ms", 0.0},
                      {"max_ms", 0.0}, {"cat", s.cat}};
            a["count"] = a["count"].get<int>() + 1;
            double ms = s.dur / 1000.0;
            a["total_ms"] = a["total_ms"].get<double>() + ms;
            a["max_ms"] = std::max(a["max_ms"].get<double>(), ms);
        }
        J summary = J::array();
        for (auto &kv : agg)
        {
            J a = kv.second;
            double t = a["total_ms"].get<double>();
            int c = a["count"].get<int>();
            a["avg_ms"] = c ? t / c : 0;
            a["pct"] = wall_us > 0 ? 100.0 * (t * 1000.0) / wall_us : 0;
            summary.push_back(a);
        }
        std::sort(summary.begin(), summary.end(),
                  [](const J &a, const J &b) { return a["total_ms"] > b["total_ms"]; });
        return J{{"source", "chrome_trace"}, {"total_ms", wall_us / 1000.0},
                 {"summary", summary}, {"spans", span_arr}, {"phases", J::object()}};
    }

    static std::string fmt(double v, int prec = 4)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        return buf;
    }

    J make_report(const J &job, const std::vector<J> &metrics, const J &grad_order,
                  const J &last_grad, const J &profile, const std::string &log_tail)
    {
        std::ostringstream md;
        md << "# TensorConsole 报表 — " << job.value("title", job.value("id", "")) << "\n\n";
        md << "| 项 | 值 |\n|---|---|\n";
        md << "| Job | `" << job.value("id", "") << "` |\n";
        md << "| 项目 | " << job.value("project_id", "") << " |\n";
        md << "| 类型 | " << job.value("type", "") << " |\n";
        md << "| 状态 | " << job.value("status", "") << " |\n";
        md << "| 退出码 | " << (job.contains("exit_code") && !job["exit_code"].is_null()
                                   ? std::to_string(job["exit_code"].get<int>()) : "-") << " |\n";
        md << "| 开始 | " << job.value("started", "") << " |\n";
        md << "| 结束 | " << job.value("ended", "") << " |\n\n";

        J final_metrics = J::object();
        if (!metrics.empty())
        {
            // merge last values per key
            for (auto &m : metrics)
            {
                J v = m.value("v", J::object());
                for (auto &kv : v.items())
                    final_metrics[kv.key()] = kv.value();
            }
            md << "## 训练指标（末值）\n\n| 指标 | 值 |\n|---|---|\n";
            for (auto &kv : final_metrics.items())
                if (kv.value().is_number())
                    md << "| " << kv.key() << " | " << fmt(kv.value().get<double>(), 6) << " |\n";
            md << "\n";
        }

        J grad_summary = J::object();
        if (last_grad.is_object() && last_grad.contains("norms"))
        {
            const auto &norms = last_grad["norms"];
            std::vector<std::pair<std::string, double>> vals;
            std::vector<std::string> zeros;
            for (auto &n : norms)
            {
                double v = n[1].get<double>();
                vals.emplace_back(n[0].get<std::string>(), v);
                if (v == 0)
                    zeros.push_back(n[0].get<std::string>());
            }
            auto sorted = vals;
            std::sort(sorted.begin(), sorted.end(),
                      [](auto &a, auto &b) { return a.second > b.second; });
            md << "## 梯度示波器摘要（step " << last_grad.value("step", 0) << "）\n\n";
            md << "参数数：" << vals.size() << "，零梯度：" << zeros.size() << "\n\n";
            md << "Top-10 梯度范数：\n\n| 参数 | grad_l2 |\n|---|---|\n";
            for (size_t i = 0; i < std::min<size_t>(10, sorted.size()); ++i)
                md << "| " << sorted[i].first << " | " << fmt(sorted[i].second, 6) << " |\n";
            if (!zeros.empty())
            {
                md << "\n零梯度参数（可能死支路/冻结）：\n";
                for (size_t i = 0; i < std::min<size_t>(20, zeros.size()); ++i)
                    md << "- " << zeros[i] << "\n";
            }
            md << "\n";
            grad_summary = J{{"step", last_grad.value("step", 0)},
                             {"n_params", vals.size()},
                             {"zero_grad", zeros},
                             {"top10", J::array()}};
            for (size_t i = 0; i < std::min<size_t>(10, sorted.size()); ++i)
                grad_summary["top10"].push_back({{"name", sorted[i].first}, {"l2", sorted[i].second}});
        }

        J prof_summary = J::array();
        if (profile.is_object() && profile.contains("summary"))
        {
            md << "## 耗时分析（" << profile.value("source", "") << "，total "
               << fmt(profile.value("total_ms", 0.0), 6) << " ms）\n\n";
            md << "| 名称 | 类别 | 次数 | 总耗时(ms) | 平均(ms) | 占比 |\n|---|---|---|---|---|---|\n";
            const auto &sum = profile["summary"];
            for (size_t i = 0; i < std::min<size_t>(15, sum.size()); ++i)
            {
                const J &s = sum[i];
                md << "| " << s.value("name", "") << " | " << s.value("kind", "") << " | "
                   << s.value("count", 0) << " | " << fmt(s.value("total_ms", 0.0), 6) << " | "
                   << fmt(s.value("avg_ms", 0.0), 4) << " | " << fmt(s.value("pct", 0.0), 4) << "% |\n";
                prof_summary.push_back(s);
            }
            md << "\n";
        }

        if (!log_tail.empty())
            md << "## 日志尾部\n\n```\n" << log_tail << "\n```\n";

        J json = J{{"job", job},
                   {"final_metrics", final_metrics},
                   {"grad_summary", grad_summary},
                   {"profile_summary", prof_summary}};
        return J{{"md", md.str()}, {"json", json}};
    }
}