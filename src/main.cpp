// tcd — TensorConsole daemon: HTTP/SSE API + job orchestration + GPU monitor.
#include "api.hpp"
#include "common.hpp"
#include "events.hpp"
#include "jobs.hpp"
#include "monitor.hpp"
#include "store.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>

static void usage()
{
    std::printf(
        "tcd — TensorConsole daemon\n"
        "usage: tcd [--bind ADDR] [--port N] [--token TOKEN] [--data DIR] [--web DIR]\n"
        "  --bind   listen address (default 127.0.0.1; non-loopback requires --token)\n"
        "  --port   listen port (default 8686)\n"
        "  --token  bearer token for /api/* (env TCD_TOKEN)\n"
        "  --data   data directory (default ./data, env TCD_DATA)\n"
        "  --web    web UI directory (default ./web, env TCD_WEB)\n");
}

int main(int argc, char **argv)
{
    using namespace tc;
    Config cfg;
    if (const char *e = std::getenv("TCD_TOKEN"))
        cfg.token = e;
    if (const char *e = std::getenv("TCD_DATA"))
        cfg.data_dir = e;
    if (const char *e = std::getenv("TCD_WEB"))
        cfg.web_dir = e;
    if (const char *e = std::getenv("TCD_PORT"))
        cfg.port = (unsigned short)std::atoi(e);

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&](const char *what) -> std::string {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "missing value for %s\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--bind")
            cfg.bind = next("--bind");
        else if (a == "--port")
            cfg.port = (unsigned short)std::stoi(next("--port"));
        else if (a == "--token")
            cfg.token = next("--token");
        else if (a == "--data")
            cfg.data_dir = next("--data");
        else if (a == "--web")
            cfg.web_dir = next("--web");
        else if (a == "--help" || a == "-h")
        {
            usage();
            return 0;
        }
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return 2;
        }
    }

    bool loopback = cfg.bind == "127.0.0.1" || cfg.bind == "localhost" || cfg.bind == "::1";
    if (!loopback && cfg.token.empty())
    {
        std::fprintf(stderr, "refusing to bind %s without --token (see docs/DESIGN.md §10)\n",
                     cfg.bind.c_str());
        return 2;
    }
    if (!fs::exists(cfg.web_dir))
        std::fprintf(stderr, "[warn] web dir not found: %s (UI disabled)\n",
                     fs::absolute(cfg.web_dir).string().c_str());

    try
    {
        ensure_dir(cfg.data_dir);
        Store store(cfg.data_dir);
        EventBus bus;
        JobManager jobs(cfg, store, bus);
        Monitor mon(bus);
        mon.set_pid_resolver([&jobs](int64_t pid) { return jobs.job_of_pid(pid); });
        jobs.recover();
        run_server(cfg, store, jobs, mon, bus);
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "[tcd] fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}