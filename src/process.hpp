#pragma once
#include "common.hpp"

#include <thread>

namespace tc
{
    struct ProcSpec
    {
        std::string exe;
        std::vector<std::string> argv; // without exe
        std::string cwd;
        std::map<std::string, std::string> env; // overrides merged onto host env
    };

    // Cross-platform child process with merged stdout/stderr line callback.
    // Windows: JobObject kills the whole tree on close/terminate.
    // POSIX: own process group, kill(-pgid).
    class Process
    {
    public:
        using LineFn = std::function<void(const std::string &)>;

        Process() = default;
        ~Process();
        Process(const Process &) = delete;
        Process &operator=(const Process &) = delete;

        bool start(const ProcSpec &spec, LineFn on_line);
        int64_t pid() const { return pid_; }
        bool running();
        std::optional<int> wait(int timeout_ms);
        std::optional<int> exit_code() const;
        void terminate(bool graceful);

    private:
        void reader_loop();

        int64_t pid_ = -1;
        std::atomic<bool> started_{false};
        std::atomic<bool> reaped_{false};
        std::atomic<int> exit_code_{-1};
        std::thread reader_;
        std::string partial_;
        LineFn on_line_;
        std::function<void(bool)> terminate_fn_;
        std::function<void()> cleanup_fn_;
#ifdef _WIN32
        void *h_process_ = nullptr;
        void *h_job_ = nullptr;
        void *h_read_ = nullptr;
#else
        int read_fd_ = -1;
#endif
    };
}