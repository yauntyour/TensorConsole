#pragma once
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace tc
{
    using J = nlohmann::json;
    namespace fs = std::filesystem;

    // ------------------------------------------------------------- time / id
    int64_t now_ms();
    std::string iso_utc(int64_t ms = -1);
    std::string new_id(const std::string &prefix);

    // ------------------------------------------------------------- strings
    std::string trim(const std::string &s);
    std::vector<std::string> split(const std::string &s, char d);
    // Whitespace tokenizer that honours "..." / '...' groups (quotes stripped).
    std::vector<std::string> split_quoted(const std::string &s);
    bool starts_with(const std::string &s, const std::string &p);
    bool ends_with(const std::string &s, const std::string &p);
    std::string to_lower(std::string s);
    std::string url_decode(const std::string &s);
    std::map<std::string, std::string> parse_query(const std::string &query);
    std::string join(const std::vector<std::string> &v, const std::string &sep);

    // ------------------------------------------------------------- filesystem
    std::string read_file(const fs::path &p);
    bool write_file_atomic(const fs::path &p, const std::string &data);
    void append_file(const fs::path &p, const std::string &data);
    void ensure_dir(const fs::path &p);
    std::vector<fs::path> list_dir(const fs::path &p);

    // ------------------------------------------------------------- config
    struct Config
    {
        std::string bind = "127.0.0.1";
        unsigned short port = 8686;
        std::string token;            // empty = no auth (loopback only)
        fs::path data_dir = "data";
        fs::path web_dir = "web";
    };

    // ------------------------------------------------------------- http
    struct Request
    {
        std::string raw;
        std::string method;
        std::string path;
        std::map<std::string, std::string> query;
        std::string body;
        std::string header(const std::string &name) const;
        std::string param(const std::string &name, const std::string &def = "") const;
        J json_body() const;
    };
    Request parse_request(const std::string &raw);

    std::string http_response(int status, const std::string &ctype, const std::string &body,
                              const std::string &extra_headers = "");
    std::string json_resp(const J &j, int status = 200);
    std::string error_resp(int status, const std::string &msg);
    std::string file_resp(const fs::path &p);
    std::string mime_of(const fs::path &p);
    const char *status_text(int code);

    // ------------------------------------------------------------- subprocess capture
    // Run a command to completion, capture merged stdout/stderr (bounded).
    struct CaptureResult
    {
        int exit_code = -1;
        std::string output;
        bool started = false;
    };
    CaptureResult run_capture(const std::string &exe, const std::vector<std::string> &argv,
                              int timeout_ms = 15000, size_t max_bytes = 4 * 1024 * 1024);

    // ------------------------------------------------------------- misc
    std::string exe_name(const std::string &base); // appends .exe on Windows
    std::vector<std::string> host_env();           // current process environment as KEY=VALUE
}