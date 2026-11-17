#pragma once
#include "common.hpp"

namespace tc
{
    class Store;
    class JobManager;
    class Monitor;
    class EventBus;

    // Registers all REST/SSE/static routes and runs the HTTP server until
    // SIGINT/SIGTERM. servic.hpp is included only in api.cpp (its router
    // header defines a non-inline free function, so it must live in one TU).
    void run_server(Config &cfg, Store &store, JobManager &jobs, Monitor &mon, EventBus &bus);
}