#pragma once
#include "common.hpp"
#include "events.hpp"

#include <deque>
#include <thread>

namespace tc
{
    class Monitor
    {
    public:
        explicit Monitor(EventBus &bus);
        ~Monitor();

        void set_pid_resolver(std::function<J(int64_t)> fn);
        void start();
        void stop();

        J gpus();
        J gpu_history();
        J system_info();

    private:
        void loop();
        J sample_gpus();
        J sample_host();

        EventBus &bus_;
        std::thread th_;
        std::atomic<bool> run_{false};
        std::mutex mx_;
        J latest_ = J::object();
        J host_ = J::object();
        std::deque<J> hist_;
        std::function<J(int64_t)> resolver_;
#ifdef _WIN32
        uint64_t prev_idle_ = 0, prev_kernel_ = 0, prev_user_ = 0;
#else
        long long prev_total_ = 0, prev_idle_ = 0;
#endif
    };
}