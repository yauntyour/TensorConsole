#include "process.hpp"

#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace tc
{
#ifdef _WIN32
    static std::wstring to_wide(const std::string &s)
    {
        if (s.empty())
            return {};
        int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
        std::wstring w(n, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
        return w;
    }

    static std::wstring quote_arg(const std::string &a)
    {
        std::wstring w = to_wide(a);
        if (!w.empty() && w.find_first_of(L" \t\"") == std::wstring::npos)
            return w;
        std::wstring out = L"\"";
        size_t bs = 0;
        for (wchar_t c : w)
        {
            if (c == L'\\')
            {
                ++bs;
                out += c;
            }
            else if (c == L'"')
            {
                out.append(bs * 2 + 1, L'\\');
                out += c;
                bs = 0;
            }
            else
            {
                bs = 0;
                out += c;
            }
        }
        out.append(bs * 2, L'\\');
        out += L'"';
        return out;
    }
#endif

    static std::vector<std::string> merged_env(const std::map<std::string, std::string> &over)
    {
        std::map<std::string, std::string> env;
        for (auto &kv : host_env())
        {
            auto eq = kv.find('=');
            if (eq != std::string::npos && eq > 0)
                env[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
        for (auto &kv : over)
            env[kv.first] = kv.second;
        std::vector<std::string> out;
        for (auto &kv : env)
            out.push_back(kv.first + "=" + kv.second);
        return out;
    }

    // POSIX 的 execve 不做 PATH 查找：`docker`、`cmake` 这类裸命令必须先解析成绝对路径，
    // 否则子进程直接 _exit(127)。带 `/` 的路径（build/train、/usr/bin/docker）原样返回。
    static std::string resolve_exe(const std::string &exe)
    {
#ifdef _WIN32
        return exe;
#else
        if (exe.empty() || exe.find('/') != std::string::npos)
            return exe;
        const char *path = std::getenv("PATH");
        if (!path)
            return exe;
        std::string p(path), dir;
        size_t i = 0;
        for (;;)
        {
            size_t j = p.find(':', i);
            dir = p.substr(i, j == std::string::npos ? std::string::npos : j - i);
            if (!dir.empty())
            {
                std::string cand = dir + "/" + exe;
                if (::access(cand.c_str(), X_OK) == 0)
                    return cand;
            }
            if (j == std::string::npos)
                break;
            i = j + 1;
        }
        return exe;
#endif
    }

    void Process::reader_loop()
    {
        char buf[8192];
        for (;;)
        {
#ifdef _WIN32
            DWORD n = 0;
            BOOL ok = ReadFile(h_read_, buf, sizeof(buf), &n, nullptr);
            if (!ok || n == 0)
                break;
#else
            ssize_t n = read(read_fd_, buf, sizeof(buf));
            if (n < 0)
            {
                if (errno == EINTR)
                    continue;
                break;
            }
            if (n == 0)
                break;
#endif
            partial_.append(buf, (size_t)n);
            size_t pos;
            while ((pos = partial_.find('\n')) != std::string::npos)
            {
                std::string line = partial_.substr(0, pos);
                partial_.erase(0, pos + 1);
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                if (on_line_)
                    on_line_(line);
            }
            if (partial_.size() > 1024 * 1024) // runaway single line
            {
                if (on_line_)
                    on_line_(partial_);
                partial_.clear();
            }
        }
        if (!partial_.empty() && on_line_)
            on_line_(partial_);
        partial_.clear();
    }

    bool Process::start(const ProcSpec &spec, LineFn on_line)
    {
        if (started_)
            return false;
        on_line_ = std::move(on_line);
        auto env = merged_env(spec.env);
        const std::string exe = resolve_exe(spec.exe);

#ifdef _WIN32
        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;
        HANDLE rd = nullptr, wr = nullptr;
        if (!CreatePipe(&rd, &wr, &sa, 0))
            return false;
        SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

        std::wstring cmd = quote_arg(spec.exe);
        for (auto &a : spec.argv)
            cmd += L" " + quote_arg(a);
        std::vector<wchar_t> cmd_buf(cmd.begin(), cmd.end());
        cmd_buf.push_back(L'\0');

        std::wstring wcwd = to_wide(spec.cwd);
        std::vector<wchar_t> env_block;
        for (auto &e : env)
        {
            auto w = to_wide(e);
            env_block.insert(env_block.end(), w.begin(), w.end());
            env_block.push_back(L'\0');
        }
        env_block.push_back(L'\0');

        // Restrict inheritance to the stdout pipe: a duplicated server socket
        // would keep HTTP/SSE connections open long after tcd replied.
        STARTUPINFOEXW si{};
        si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        si.StartupInfo.hStdOutput = wr;
        si.StartupInfo.hStdError = wr;
        si.StartupInfo.hStdInput = nullptr;
        bool listed = false, extended = false;
        SIZE_T attr_size = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
        std::vector<char> attr_buf(attr_size > 0 ? attr_size : 1);
        si.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attr_buf.data());
        HANDLE inherit[] = {wr};
        if (attr_size > 0 &&
            InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &attr_size))
        {
            listed = true;
            extended = UpdateProcThreadAttribute(si.lpAttributeList, 0,
                                                 PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                                 inherit, sizeof(inherit), nullptr, nullptr);
        }
        si.StartupInfo.cb = extended ? sizeof(STARTUPINFOEXW) : sizeof(STARTUPINFO);
        PROCESS_INFORMATION pi{};
        DWORD flags = CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW;
        if (extended)
            flags |= EXTENDED_STARTUPINFO_PRESENT;
        BOOL ok = CreateProcessW(nullptr, cmd_buf.data(), nullptr, nullptr, TRUE, flags,
                                 env_block.data(), wcwd.empty() ? nullptr : wcwd.c_str(),
                                 &si.StartupInfo, &pi);
        if (listed)
            DeleteProcThreadAttributeList(si.lpAttributeList);
        CloseHandle(wr);
        if (!ok)
        {
            CloseHandle(rd);
            return false;
        }
        h_read_ = rd;
        h_process_ = pi.hProcess;
        pid_ = (int64_t)pi.dwProcessId;
        CloseHandle(pi.hThread);

        h_job_ = CreateJobObjectW(nullptr, nullptr);
        if (h_job_)
        {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
            li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(h_job_, JobObjectExtendedLimitInformation, &li, sizeof(li));
            AssignProcessToJobObject(h_job_, pi.hProcess);
        }

        terminate_fn_ = [this](bool graceful) {
            (void)graceful; // no console-ctrl for detached children; JobManager uses stop-files
            if (h_process_)
                TerminateProcess(h_process_, 1);
        };
        cleanup_fn_ = [this] {
            if (h_job_)
            {
                CloseHandle(h_job_); // KILL_ON_JOB_CLOSE reaps the tree
                h_job_ = nullptr;
            }
            if (h_process_)
            {
                CloseHandle(h_process_);
                h_process_ = nullptr;
            }
            if (h_read_)
            {
                CloseHandle(h_read_);
                h_read_ = nullptr;
            }
        };
#else
        int fds[2];
        if (pipe(fds) != 0)
            return false;
        pid_t child = fork();
        if (child < 0)
        {
            close(fds[0]);
            close(fds[1]);
            return false;
        }
        if (child == 0)
        {
            setpgid(0, 0);
            close(fds[0]);
            dup2(fds[1], STDOUT_FILENO);
            dup2(fds[1], STDERR_FILENO);
            close(fds[1]);
            if (!spec.cwd.empty() && chdir(spec.cwd.c_str()) != 0)
                _exit(127);
            std::vector<std::string> args;
            args.push_back(exe);
            for (auto &a : spec.argv)
                args.push_back(a);
            std::vector<char *> argv;
            for (auto &a : args)
                argv.push_back(const_cast<char *>(a.c_str()));
            argv.push_back(nullptr);
            std::vector<char *> envp;
            for (auto &e : env)
                envp.push_back(const_cast<char *>(e.c_str()));
            envp.push_back(nullptr);
            execve(exe.c_str(), argv.data(), envp.data());
            _exit(127);
        }
        close(fds[1]);
        read_fd_ = fds[0];
        pid_ = child;
        terminate_fn_ = [this](bool graceful) {
            if (pid_ > 0)
                kill((pid_t)-pid_, graceful ? SIGINT : SIGKILL);
        };
        cleanup_fn_ = [this] {
            if (read_fd_ >= 0)
            {
                close(read_fd_);
                read_fd_ = -1;
            }
        };
#endif
        started_ = true;
        reader_ = std::thread([this] { reader_loop(); });
        return true;
    }

    bool Process::running()
    {
        if (!started_ || reaped_)
            return false;
#ifdef _WIN32
        DWORD rc = WaitForSingleObject(h_process_, 0);
        if (rc == WAIT_OBJECT_0)
        {
            DWORD code = 0;
            GetExitCodeProcess(h_process_, &code);
            exit_code_ = (int)code;
            reaped_ = true;
            return false;
        }
        return true;
#else
        int st = 0;
        pid_t r = waitpid((pid_t)pid_, &st, WNOHANG);
        if (r == pid_)
        {
            exit_code_ = WIFEXITED(st) ? WEXITSTATUS(st) : (128 + WTERMSIG(st));
            reaped_ = true;
            return false;
        }
        return r == 0;
#endif
    }

    std::optional<int> Process::wait(int timeout_ms)
    {
        if (!started_)
            return std::nullopt;
        if (reaped_)
            return exit_code_.load();
#ifdef _WIN32
        DWORD rc = WaitForSingleObject(h_process_, timeout_ms < 0 ? INFINITE : (DWORD)timeout_ms);
        if (rc != WAIT_OBJECT_0)
            return std::nullopt;
        DWORD code = 0;
        GetExitCodeProcess(h_process_, &code);
        exit_code_ = (int)code;
        reaped_ = true;
        return exit_code_.load();
#else
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;)
        {
            int st = 0;
            pid_t r = waitpid((pid_t)pid_, &st, WNOHANG);
            if (r == pid_)
            {
                exit_code_ = WIFEXITED(st) ? WEXITSTATUS(st) : (128 + WTERMSIG(st));
                reaped_ = true;
                return exit_code_.load();
            }
            if (r < 0)
            {
                reaped_ = true;
                return exit_code_.load();
            }
            if (timeout_ms >= 0 && std::chrono::steady_clock::now() >= deadline)
                return std::nullopt;
            usleep(20000);
        }
#endif
    }

    std::optional<int> Process::exit_code() const
    {
        if (reaped_)
            return exit_code_.load();
        return std::nullopt;
    }

    void Process::terminate(bool graceful)
    {
        if (started_ && terminate_fn_)
            terminate_fn_(graceful);
    }

    Process::~Process()
    {
        if (started_)
        {
            if (running())
                terminate(false);
            wait(3000);
            if (reader_.joinable())
                reader_.join();
            if (cleanup_fn_)
                cleanup_fn_();
        }
    }

    CaptureResult run_capture(const std::string &exe, const std::vector<std::string> &argv,
                              int timeout_ms, size_t max_bytes)
    {
        CaptureResult res;
        std::mutex mx;
        {
            ProcSpec spec;
            spec.exe = exe;
            spec.argv = argv;
            Process p;
            res.started = p.start(spec, [&](const std::string &line) {
                std::lock_guard<std::mutex> lk(mx);
                if (res.output.size() < max_bytes)
                    res.output += line + "\n";
            });
            if (!res.started)
                return res;
            auto code = p.wait(timeout_ms);
            if (!code)
            {
                p.terminate(false);
                code = p.wait(3000);
            }
            res.exit_code = code.value_or(-1);
        } // ~Process joins the reader thread, flushing remaining output into res
        return res;
    }
}