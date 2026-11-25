#include "common.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
extern char **environ;
#endif

namespace tc
{
    int64_t now_ms()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    std::string iso_utc(int64_t ms)
    {
        if (ms < 0)
            ms = now_ms();
        std::time_t t = (std::time_t)(ms / 1000);
        std::tm tm{};
#ifdef _WIN32
        gmtime_s(&tm, &t);
#else
        gmtime_r(&t, &tm);
#endif
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ", tm.tm_year + 1900,
                      tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
        return buf;
    }

    std::string new_id(const std::string &prefix)
    {
        static std::mt19937_64 rng{std::random_device{}()};
        std::time_t t = (std::time_t)(now_ms() / 1000);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s-%04d%02d%02d-%02d%02d%02d-%04x", prefix.c_str(),
                      tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec,
                      (unsigned)(rng() & 0xFFFF));
        return buf;
    }

    std::string trim(const std::string &s)
    {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos)
            return "";
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    }

    std::vector<std::string> split(const std::string &s, char d)
    {
        std::vector<std::string> out;
        size_t start = 0;
        for (size_t i = 0; i <= s.size(); ++i)
            if (i == s.size() || s[i] == d)
            {
                out.push_back(s.substr(start, i - start));
                start = i + 1;
            }
        return out;
    }

    std::vector<std::string> split_quoted(const std::string &s)
    {
        std::vector<std::string> out;
        std::string cur;
        char quote = 0;
        bool has = false;
        auto flush = [&] {
            if (has || !cur.empty())
            {
                out.push_back(cur);
                cur.clear();
                has = false;
            }
        };
        for (char c : s)
        {
            if (quote)
            {
                if (c == quote)
                    quote = 0;
                else
                    cur += c;
                continue;
            }
            if (c == '"' || c == '\'')
            {
                quote = c;
                has = true;
                continue;
            }
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
            {
                flush();
                continue;
            }
            cur += c;
            has = true;
        }
        flush();
        return out;
    }

    bool starts_with(const std::string &s, const std::string &p)
    {
        return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
    }

    bool ends_with(const std::string &s, const std::string &p)
    {
        return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
    }

    std::string to_lower(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        return s;
    }

    std::string url_decode(const std::string &s)
    {
        std::string out;
        for (size_t i = 0; i < s.size(); ++i)
        {
            if (s[i] == '%' && i + 2 < s.size())
            {
                auto hex = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                    return -1;
                };
                int h = hex(s[i + 1]), l = hex(s[i + 2]);
                if (h >= 0 && l >= 0)
                {
                    out += (char)(h * 16 + l);
                    i += 2;
                    continue;
                }
            }
            out += s[i] == '+' ? ' ' : s[i];
        }
        return out;
    }

    std::map<std::string, std::string> parse_query(const std::string &query)
    {
        std::map<std::string, std::string> out;
        for (auto &kv : split(query, '&'))
        {
            if (kv.empty())
                continue;
            auto eq = kv.find('=');
            if (eq == std::string::npos)
                out[url_decode(kv)] = "";
            else
                out[url_decode(kv.substr(0, eq))] = url_decode(kv.substr(eq + 1));
        }
        return out;
    }

    std::string join(const std::vector<std::string> &v, const std::string &sep)
    {
        std::string out;
        for (size_t i = 0; i < v.size(); ++i)
        {
            if (i)
                out += sep;
            out += v[i];
        }
        return out;
    }

    std::string read_file(const fs::path &p)
    {
        std::ifstream f(p, std::ios::binary);
        if (!f)
            return "";
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    bool write_file_atomic(const fs::path &p, const std::string &data)
    {
        ensure_dir(p.parent_path());
        fs::path tmp = p;
        tmp += ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f)
                return false;
            f.write(data.data(), (std::streamsize)data.size());
        }
        std::error_code ec;
        fs::rename(tmp, p, ec);
        if (ec)
        {
            fs::remove(p, ec);
            fs::rename(tmp, p, ec);
        }
        return !ec;
    }

    void append_file(const fs::path &p, const std::string &data)
    {
        ensure_dir(p.parent_path());
        std::ofstream f(p, std::ios::binary | std::ios::app);
        if (f)
            f.write(data.data(), (std::streamsize)data.size());
    }

    void ensure_dir(const fs::path &p)
    {
        if (!p.empty())
        {
            std::error_code ec;
            fs::create_directories(p, ec);
        }
    }

    std::vector<fs::path> list_dir(const fs::path &p)
    {
        std::vector<fs::path> out;
        std::error_code ec;
        if (fs::is_directory(p, ec))
            for (auto &e : fs::directory_iterator(p, ec))
                out.push_back(e.path());
        std::sort(out.begin(), out.end());
        return out;
    }

    // ------------------------------------------------------------- http

    std::string Request::header(const std::string &name) const
    {
        // case-insensitive scan of the header block
        auto head_end = raw.find("\r\n\r\n");
        std::string block = raw.substr(0, head_end == std::string::npos ? raw.size() : head_end);
        std::string want = to_lower(name);
        for (auto &line : split(block, '\n'))
        {
            auto l = trim(line);
            auto colon = l.find(':');
            if (colon == std::string::npos)
                continue;
            if (to_lower(trim(l.substr(0, colon))) == want)
                return trim(l.substr(colon + 1));
        }
        return "";
    }

    std::string Request::param(const std::string &name, const std::string &def) const
    {
        auto it = query.find(name);
        return it == query.end() ? def : it->second;
    }

    J Request::json_body() const
    {
        if (trim(body).empty())
            return J::object();
        return J::parse(body, nullptr, false);
    }

    Request parse_request(const std::string &raw)
    {
        Request r;
        r.raw = raw;
        auto first_end = raw.find_first_of("\r\n");
        std::string first = raw.substr(0, first_end);
        std::istringstream iss(first);
        std::string url, version;
        iss >> r.method >> url >> version;
        auto q = url.find('?');
        if (q == std::string::npos)
            r.path = url_decode(url);
        else
        {
            r.path = url_decode(url.substr(0, q));
            r.query = parse_query(url.substr(q + 1));
        }
        auto head_end = raw.find("\r\n\r\n");
        if (head_end != std::string::npos)
            r.body = raw.substr(head_end + 4);
        return r;
    }

    const char *status_text(int code)
    {
        switch (code)
        {
        case 200: return "OK";
        case 201: return "Created";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 409: return "Conflict";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default: return "OK";
        }
    }

    std::string http_response(int status, const std::string &ctype, const std::string &body,
                              const std::string &extra_headers)
    {
        std::ostringstream ss;
        ss << "HTTP/1.1 " << status << " " << status_text(status) << "\r\n"
           << "Content-Type: " << ctype << "\r\n"
           << "Content-Length: " << body.size() << "\r\n"
           << "Access-Control-Allow-Origin: *\r\n"
           << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
           << "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
           << "Connection: close\r\n"
           << extra_headers << "\r\n"
           << body;
        return ss.str();
    }

    std::string json_resp(const J &j, int status)
    {
        return http_response(status, "application/json; charset=utf-8", j.dump());
    }

    std::string error_resp(int status, const std::string &msg)
    {
        return json_resp(J{{"error", msg}, {"code", status}}, status);
    }

    std::string mime_of(const fs::path &p)
    {
        auto ext = to_lower(p.extension().string());
        if (ext == ".html") return "text/html; charset=utf-8";
        if (ext == ".js") return "text/javascript; charset=utf-8";
        if (ext == ".css") return "text/css; charset=utf-8";
        if (ext == ".json") return "application/json; charset=utf-8";
        if (ext == ".svg") return "image/svg+xml";
        if (ext == ".png") return "image/png";
        if (ext == ".ico") return "image/x-icon";
        if (ext == ".csv") return "text/csv; charset=utf-8";
        if (ext == ".md") return "text/markdown; charset=utf-8";
        if (ext == ".woff2") return "font/woff2";
        return "application/octet-stream";
    }

    std::string file_resp(const fs::path &p)
    {
        std::error_code ec;
        if (!fs::is_regular_file(p, ec))
            return error_resp(404, "file not found: " + p.filename().string());
        std::string data = read_file(p);
        return http_response(200, mime_of(p), data);
    }

    std::string exe_name(const std::string &base)
    {
#ifdef _WIN32
        return base + ".exe";
#else
        return base;
#endif
    }

    std::vector<std::string> host_env()
    {
        std::vector<std::string> out;
#ifdef _WIN32
        LPWCH env = GetEnvironmentStringsW();
        if (env)
        {
            for (LPWCH p = env; *p;)
            {
                int len = WideCharToMultiByte(CP_UTF8, 0, p, -1, nullptr, 0, nullptr, nullptr);
                std::string s(len > 0 ? len - 1 : 0, '\0');
                if (len > 1)
                    WideCharToMultiByte(CP_UTF8, 0, p, -1, s.data(), len, nullptr, nullptr);
                out.push_back(s);
                p += wcslen(p) + 1;
            }
            FreeEnvironmentStringsW(env);
        }
#else
        for (char **e = environ; *e; ++e)
            out.push_back(*e);
#endif
        return out;
    }
}