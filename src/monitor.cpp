#include "monitor.hpp"

#include <algorithm>
#include <fstream>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace tc
{
    static double csv_num(const std::string &s)
    {
        auto t = trim(s);
        if (t.empty() || t[0] == '[')
            return -1;
        try
        {
            return std::stod(t);
        }
        catch (...)
        {
            return -1;
        }
    }

    // Cluster-wide roll-up of per-card samples. Metrics reported as -1 (N/A) are
    // excluded from the aggregates so one unsupported card cannot skew the mean.
    static J cluster_summary(const J &gpus)
    {
        int n = 0, n_util = 0;
        double util_sum = 0, util_max = -1, temp_max = -1;
        double mem_used = 0, mem_total = 0, power = 0;
        bool has_mem = false, has_power = false;
        for (const auto &g : gpus)
        {
            ++n;
            double u = g.value("util", -1.0);
            if (u >= 0)
            {
                util_sum += u;
                ++n_util;
                util_max = std::max(util_max, u);
            }
            double mu = g.value("mem_used", -1.0), mt = g.value("mem_total", -1.0);
            if (mu >= 0)
            {
                mem_used += mu;
                has_mem = true;
            }
            if (mt >= 0)
            {
                mem_total += mt;
                has_mem = true;
            }
            double p = g.value("power", -1.0);
            if (p >= 0)
            {
                power += p;
                has_power = true;
            }
            temp_max = std::max(temp_max, g.value("temp", -1.0));
        }
        return J{{"count", n},
                 {"util", n_util ? util_sum / n_util : -1.0},
                 {"util_max", util_max},
                 {"mem_used", has_mem ? mem_used : -1.0},
                 {"mem_total", has_mem ? mem_total : -1.0},
                 {"mem_pct", mem_total > 0 ? 100.0 * mem_used / mem_total : -1.0},
                 {"power", has_power ? power : -1.0},
                 {"temp_max", temp_max}};
    }

    Monitor::Monitor(EventBus &bus) : bus_(bus) {}
    Monitor::~Monitor() { stop(); }

    void Monitor::set_pid_resolver(std::function<J(int64_t)> fn) { resolver_ = std::move(fn); }

    void Monitor::start()
    {
        if (run_)
            return;
        run_ = true;
        th_ = std::thread([this] { loop(); });
    }

    void Monitor::stop()
    {
        run_ = false;
        if (th_.joinable())
            th_.join();
    }

    J Monitor::sample_gpus()
    {
        J gpus = J::array();
        auto r = run_capture("nvidia-smi",
                             {"--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,"
                              "temperature.gpu,power.draw",
                              "--format=csv,noheader,nounits"},
                             6000);
        if (!r.started || r.exit_code != 0)
            return gpus;
        std::map<std::string, int> uuid_idx;
        for (auto &line : split(r.output, '\n'))
        {
            auto c = split(line, ',');
            if (c.size() < 8)
                continue;
            int idx = (int)csv_num(c[0]);
            if (idx < 0)
                continue;
            uuid_idx[trim(c[1])] = idx;
            gpus.push_back({{"index", idx},
                            {"name", trim(c[2])},
                            {"util", csv_num(c[3])},
                            {"mem_used", csv_num(c[4])},
                            {"mem_total", csv_num(c[5])},
                            {"temp", csv_num(c[6])},
                            {"power", csv_num(c[7])},
                            {"processes", J::array()}});
        }
        auto apps = run_capture("nvidia-smi",
                                {"--query-compute-apps=pid,gpu_uuid,used_memory",
                                 "--format=csv,noheader,nounits"},
                                6000);
        if (apps.started && apps.exit_code == 0)
        {
            for (auto &line : split(apps.output, '\n'))
            {
                auto c = split(line, ',');
                if (c.size() < 3)
                    continue;
                int64_t pid = (int64_t)csv_num(c[0]);
                auto it = uuid_idx.find(trim(c[1]));
                if (pid <= 0 || it == uuid_idx.end())
                    continue;
                J p = {{"pid", pid}, {"used_memory", csv_num(c[2])}};
                if (resolver_)
                {
                    J j = resolver_(pid);
                    if (j.is_object())
                        p["job"] = j;
                }
                for (auto &g : gpus)
                    if (g["index"].get<int>() == it->second)
                        g["processes"].push_back(p);
            }
        }
        return gpus;
    }

    J Monitor::sample_host()
    {
        J h = J::object();
#ifdef _WIN32
        FILETIME idle, kernel, user;
        if (GetSystemTimes(&idle, &kernel, &user))
        {
            auto ft = [](FILETIME f) -> uint64_t {
                return ((uint64_t)f.dwHighDateTime << 32) | f.dwLowDateTime;
            };
            uint64_t i = ft(idle), k = ft(kernel), u = ft(user);
            uint64_t di = i - prev_idle_, dk = k - prev_kernel_, du = u - prev_user_;
            uint64_t tot = dk + du;
            if (prev_kernel_ && tot > 0)
                h["cpu_pct"] = 100.0 * (double)(tot - di) / (double)tot;
            prev_idle_ = i;
            prev_kernel_ = k;
            prev_user_ = u;
        }
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms))
        {
            h["mem_total_mb"] = (double)(ms.ullTotalPhys / 1048576);
            h["mem_used_mb"] = (double)((ms.ullTotalPhys - ms.ullAvailPhys) / 1048576);
            h["mem_pct"] = (double)ms.dwMemoryLoad;
        }
#else
        std::ifstream stat("/proc/stat");
        std::string line;
        if (std::getline(stat, line))
        {
            long long u = 0, n = 0, s = 0, i = 0, w = 0, irq = 0, sirq = 0, st = 0;
            if (std::sscanf(line.c_str(), "cpu %lld %lld %lld %lld %lld %lld %lld %lld",
                            &u, &n, &s, &i, &w, &irq, &sirq, &st) >= 4)
            {
                long long total = u + n + s + i + w + irq + sirq + st;
                long long idle = i + w;
                if (prev_total_ && total > prev_total_)
                    h["cpu_pct"] = 100.0 * (double)((total - prev_total_) - (idle - prev_idle_)) /
                                   (double)(total - prev_total_);
                prev_total_ = total;
                prev_idle_ = idle;
            }
        }
        std::ifstream mem("/proc/meminfo");
        long long mt = 0, ma = 0;
        while (std::getline(mem, line))
        {
            if (starts_with(line, "MemTotal:"))
                std::sscanf(line.c_str(), "MemTotal: %lld", &mt);
            else if (starts_with(line, "MemAvailable:"))
                std::sscanf(line.c_str(), "MemAvailable: %lld", &ma);
        }
        if (mt > 0)
        {
            h["mem_total_mb"] = (double)(mt / 1024);
            h["mem_used_mb"] = (double)((mt - ma) / 1024);
            h["mem_pct"] = 100.0 * (double)(mt - ma) / (double)mt;
        }
#endif
        h["ts"] = now_ms();
        return h;
    }

    void Monitor::loop()
    {
        while (run_)
        {
            J gpus = sample_gpus();
            J host = sample_host();
            J sample;
            {
                std::lock_guard<std::mutex> lk(mx_);
                latest_ = J{{"gpus", gpus}, {"host", host}, {"cluster", cluster_summary(gpus)}};
                host_ = host;
                hist_.push_back(latest_);
                if (hist_.size() > 300)
                    hist_.pop_front();
                sample = latest_;
            }
            bus_.publish("gpu", sample);
            bus_.publish("heartbeat", J{{"ts", now_ms()}});
            for (int i = 0; i < 10 && run_; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    J Monitor::gpus()
    {
        std::lock_guard<std::mutex> lk(mx_);
        return latest_;
    }

    J Monitor::gpu_history()
    {
        std::lock_guard<std::mutex> lk(mx_);
        J out = J::array();
        for (auto &h : hist_)
            out.push_back(h);
        return out;
    }

    J Monitor::system_info()
    {
        char host[256] = {0};
        gethostname(host, sizeof(host) - 1);
        J info = {{"hostname", host},
                  {"os",
#ifdef _WIN32
                   "windows"
#else
                   "linux"
#endif
                  },
                  {"cpu_cores", (int)std::thread::hardware_concurrency()},
                  {"version", "1.0.0"}};
        std::lock_guard<std::mutex> lk(mx_);
        info["host"] = host_;
        return info;
    }
}