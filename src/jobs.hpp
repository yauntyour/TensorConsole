#pragma once
#include "common.hpp"
#include "events.hpp"
#include "process.hpp"
#include "store.hpp"

#include <thread>

namespace tc
{
    class JobManager
    {
    public:
        JobManager(Config &cfg, Store &store, EventBus &bus);
        ~JobManager();

        J submit(const J &body);
        J stop(const std::string &id);
        J job_of_pid(int64_t pid);
        void recover();

        std::vector<J> metrics(const std::string &id, const std::string &keys, int64_t from,
                               int64_t to, int64_t rank, size_t limit);
        J grad_order(const std::string &id);
        std::vector<J> grad_steps(const std::string &id, int64_t step, int64_t from, int64_t to,
                                  int64_t rank, size_t limit);
        J profile(const std::string &id);
        void upload_profile(const std::string &id, const J &trace);
        J report(const std::string &id, const std::string &format);
        J logs(const std::string &id, size_t offset, size_t limit);

    private:
        struct Rank
        {
            int rank = 0, gpu = 0;
            std::string out;
            std::unique_ptr<Process> proc;
        };
        struct Tail
        {
            std::streamoff csv_off = 0, grad_off = 0;
            std::string csv_part, grad_part;
            int64_t cur_step = -1;
            std::vector<std::pair<std::string, double>> pend;
        };
        struct Live
        {
            J rec;
            std::vector<Rank> ranks;
            std::vector<ProcSpec> phases;
            size_t phase_idx = 0;
            std::unique_ptr<Process> phase_proc;
            std::vector<Tail> tails;
            std::thread watcher;
            std::atomic<bool> stop_req{false};
            std::atomic<bool> done{false};
            int64_t stop_deadline = 0;
            std::streamoff met_off = 0;
            std::string met_part;
            J order;
            bool order_set = false;
            int64_t last_stdout_step = -1;
            std::vector<J> prof_entries;
            J prof_totals = J::object();
            std::string prof_phase;
            std::mutex mx;
        };

        Live *live(const std::string &id);
        void start_watcher(Live *L);
        void watch_train(Live *L);
        void watch_phases(Live *L);
        void on_line(Live *L, int rank, const std::string &line);
        void tail_csv(Live *L, size_t r);
        void tail_grad(Live *L, size_t r);
        void tail_metrics(Live *L);
        void flush_grad(Live *L, size_t r, int64_t step);
        void ensure_order(Live *L, const std::vector<std::string> &names);
        void finalize(Live *L, const std::string &status, int exit_code);
        void ensure_train_slot(const std::string &project_id, const J &params);
        ProcSpec build_spec(const J &proj, const J &params, std::vector<ProcSpec> &phases);
        std::vector<ProcSpec> docker_phases(const J &proj, const J &params);
        std::vector<ProcSpec> deploy_phases(const J &proj, const J &params);
        static std::map<std::string, std::string> kv_env(const std::string &s);

        Config &cfg_;
        Store &store_;
        EventBus &bus_;
        std::mutex mx_;
        std::map<std::string, std::unique_ptr<Live>> live_;
    };
}