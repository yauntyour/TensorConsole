// tc — TensorConsole CLI for humans and Agent systems.
// Talks to tcd over HTTP/SSE; every command can emit machine-readable JSON.
#include <nlohmann/json.hpp>

#include <boost/asio.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using J = nlohmann::json;
namespace asio = boost::asio;
using tcp = asio::ip::tcp;

namespace
{
    struct Server
    {
        std::string host = "127.0.0.1";
        std::string port = "8686";
        std::string token;
        bool compact = false;
    };

    void die(const std::string &msg)
    {
        std::fprintf(stderr, "tc: %s\n", msg.c_str());
        std::exit(1);
    }

    std::string read_all(const std::string &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            die("cannot read file: " + path);
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }

    struct Resp
    {
        int status = 0;
        std::string body;
    };

    std::string auth_query(const Server &srv, const std::string &path)
    {
        if (srv.token.empty())
            return path;
        return path + (path.find('?') == std::string::npos ? "?" : "&") + "token=" + srv.token;
    }

    tcp::socket connect_send(const Server &srv, const std::string &method, const std::string &path,
                             const std::string &body, const std::string &accept, asio::io_context &io)
    {
        tcp::resolver res(io);
        auto eps = res.resolve(srv.host, srv.port);
        tcp::socket sock(io);
        asio::connect(sock, eps);
        std::ostringstream rq;
        rq << method << " " << path << " HTTP/1.1\r\n"
           << "Host: " << srv.host << ":" << srv.port << "\r\n"
           << "Connection: close\r\n"
           << "Accept: " << accept << "\r\n";
        if (!srv.token.empty())
            rq << "Authorization: Bearer " << srv.token << "\r\n";
        if (!body.empty())
            rq << "Content-Type: application/json\r\nContent-Length: " << body.size() << "\r\n";
        rq << "\r\n" << body;
        asio::write(sock, asio::buffer(rq.str()));
        return sock;
    }

    Resp http(const Server &srv, const std::string &method, const std::string &path,
              const std::string &body = "")
    {
        asio::io_context io;
        auto sock = connect_send(srv, method, path, body, "application/json", io);
        std::string raw;
        char buf[16384];
        boost::system::error_code ec;
        for (;;)
        {
            size_t n = sock.read_some(asio::buffer(buf), ec);
            if (n)
                raw.append(buf, n);
            if (ec)
                break;
        }
        Resp r;
        auto sp = raw.find(' ');
        if (sp != std::string::npos)
            r.status = std::atoi(raw.c_str() + sp + 1);
        auto hb = raw.find("\r\n\r\n");
        r.body = hb == std::string::npos ? "" : raw.substr(hb + 4);
        return r;
    }

    J call(const Server &srv, const std::string &method, const std::string &path,
           const std::string &body = "")
    {
        auto r = http(srv, method, path, body);
        if (r.status < 200 || r.status >= 300)
            die(method + " " + path + " -> HTTP " + std::to_string(r.status) + ": " + r.body);
        if (r.body.empty())
            return J::object();
        auto j = J::parse(r.body, nullptr, false);
        if (j.is_discarded())
            die("invalid JSON response from " + path);
        return j;
    }

    void print_j(const Server &srv, const J &j)
    {
        std::cout << (srv.compact ? j.dump() : j.dump(2)) << "\n";
    }

    // Stream SSE events; callback returns false to stop.
    void sse(const Server &srv, const std::string &topics,
             const std::function<bool(const std::string &, const J &)> &cb)
    {
        asio::io_context io;
        std::string path = "/api/events";
        if (!topics.empty())
            path += "?topics=" + topics;
        auto sock = connect_send(srv, "GET", auth_query(srv, path), "", "text/event-stream", io);
        std::string acc, ev;
        char buf[16384];
        boost::system::error_code ec;
        bool headers_done = false;
        for (;;)
        {
            size_t n = sock.read_some(asio::buffer(buf), ec);
            if (n)
                acc.append(buf, n);
            if (!headers_done)
            {
                auto hb = acc.find("\r\n\r\n");
                if (hb == std::string::npos)
                {
                    if (ec)
                        return;
                    continue;
                }
                acc.erase(0, hb + 4);
                headers_done = true;
            }
            size_t pos;
            while ((pos = acc.find('\n')) != std::string::npos)
            {
                std::string line = acc.substr(0, pos);
                acc.erase(0, pos + 1);
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                if (line.rfind("event: ", 0) == 0)
                    ev = line.substr(7);
                else if (line.rfind("data: ", 0) == 0)
                {
                    auto d = J::parse(line.substr(6), nullptr, false);
                    if (!d.is_discarded() && !cb(ev, d))
                        return;
                }
            }
            if (ec)
                return;
        }
    }

    // ------------------------------------------------------------- args
    struct Args
    {
        std::vector<std::string> pos;
        std::map<std::string, std::string> opt;
        bool has(const std::string &k) const { return opt.count(k) > 0; }
        std::string get(const std::string &k, const std::string &def = "") const
        {
            auto it = opt.find(k);
            return it == opt.end() ? def : it->second;
        }
    };

    Args parse_args(int argc, char **argv, int start)
    {
        Args a;
        for (int i = start; i < argc; ++i)
        {
            std::string s = argv[i];
            if (s.rfind("--", 0) == 0)
            {
                std::string key = s.substr(2);
                if (i + 1 < argc && argv[i + 1][0] != '-')
                    a.opt[key] = argv[++i];
                else
                    a.opt[key] = "true";
            }
            else if (s == "-p" && i + 1 < argc)
            {
                const int k = i;
                a.opt["p:" + std::to_string(k)] = argv[++i];
            }
            else
                a.pos.push_back(s);
        }
        return a;
    }

    J typed_value(const std::string &s)
    {
        if (s == "true")
            return true;
        if (s == "false")
            return false;
        if (!s.empty() && (s[0] == '{' || s[0] == '['))
        {
            auto j = J::parse(s, nullptr, false);
            if (!j.is_discarded())
                return j;
        }
        try
        {
            size_t used = 0;
            long long v = std::stoll(s, &used);
            if (used == s.size())
                return v;
        }
        catch (...)
        {
        }
        try
        {
            size_t used = 0;
            double v = std::stod(s, &used);
            if (used == s.size())
                return v;
        }
        catch (...)
        {
        }
        return s;
    }

    J collect_params(const Args &a)
    {
        J params = J::object();
        if (a.has("params"))
        {
            std::string s = a.get("params");
            if (!s.empty() && s[0] == '@')
                s = read_all(s.substr(1));
            auto j = J::parse(s, nullptr, false);
            if (!j.is_object())
                die("--params must be a JSON object (or @file.json)");
            params = j;
        }
        for (auto &kv : a.opt)
        {
            if (kv.first.rfind("p:", 0) != 0)
                continue;
            auto eq = kv.second.find('=');
            if (eq == std::string::npos)
                die("-p expects key=value, got: " + kv.second);
            params[kv.second.substr(0, eq)] = typed_value(kv.second.substr(eq + 1));
        }
        return params;
    }

    std::string q(const std::string &s)
    {
        std::string out;
        for (char c : s)
        {
            if (isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.' || c == ',')
                out += c;
            else
            {
                char buf[8];
                std::snprintf(buf, sizeof buf, "%%%02X", (unsigned char)c);
                out += buf;
            }
        }
        return out;
    }

    bool terminal(const std::string &status)
    {
        return status == "succeeded" || status == "failed" || status == "stopped";
    }

    int wait_job(const Server &srv, const std::string &id)
    {
        std::string last;
        for (;;)
        {
            J job = call(srv, "GET", "/api/jobs/" + id);
            std::string st = job.value("status", "");
            if (st != last)
            {
                std::fprintf(stderr, "[tc] job %s: %s\n", id.c_str(), st.c_str());
                last = st;
            }
            if (terminal(st))
            {
                auto r = http(srv, "GET", "/api/jobs/" + id + "/report?format=md");
                if (r.status == 200)
                    std::fprintf(stderr, "%s\n", r.body.c_str());
                print_j(srv, job);
                return st == "succeeded" ? 0 : 1;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    void follow_logs(const Server &srv, const std::string &id)
    {
        sse(srv, "job", [&](const std::string &ev, const J &d) {
            if (d.value("job_id", "") != id)
                return true;
            if (ev == "job.log")
                std::printf("%s\n", d.value("line", "").c_str());
            else if (ev == "job.status")
            {
                std::fprintf(stderr, "[tc] status: %s\n", d.value("status", "").c_str());
                return !terminal(d.value("status", ""));
            }
            return true;
        });
    }

    int do_submit(const Server &srv, const Args &a, const std::string &type)
    {
        if (a.pos.size() < 1)
            die("usage: tc " + type + " <project> [-p key=value ...] [--wait] [--follow]");
        J body = {{"project_id", a.pos[0]}, {"type", type}, {"params", collect_params(a)}};
        if (a.has("title"))
            body["title"] = a.get("title");
        J rec = call(srv, "POST", "/api/jobs", body.dump());
        std::string id = rec.at("id").get<std::string>();
        if (!a.has("wait") && !a.has("follow"))
        {
            print_j(srv, rec);
            return 0;
        }
        if (a.has("follow"))
            follow_logs(srv, id);
        return wait_job(srv, id);
    }

    void csv_metrics(const J &rows)
    {
        std::vector<std::string> keys;
        for (auto &r : rows)
        {
            J v = r.value("v", J::object());
            for (auto &kv : v.items())
                if (std::find(keys.begin(), keys.end(), kv.key()) == keys.end())
                    keys.push_back(kv.key());
        }
        std::cout << "step,rank";
        for (auto &k : keys)
            std::cout << "," << k;
        std::cout << "\n";
        for (auto &r : rows)
        {
            std::cout << r.value("step", (int64_t)0) << "," << r.value("rank", (int64_t)0);
            for (auto &k : keys)
            {
                auto &v = r["v"];
                std::cout << ",";
                if (v.contains(k))
                    std::cout << v[k].dump();
            }
            std::cout << "\n";
        }
    }

    void usage()
    {
        std::printf(
            "tc — TensorConsole CLI (server: --server URL, auth: --token T, env TC_SERVER/TC_TOKEN)\n"
            "  health | system | gpus\n"
            "  projects list | projects add --root DIR [--id ID] [--name N] [--kind metamodel]\n"
            "  projects rm <id>\n"
            "  schema <project> [--type train|build|eval|generate|test|docker|deploy|custom]\n"
            "  submit <project> <type> [-p k=v ...] [--params JSON|@file] [--title T] [--wait] [--follow]\n"
            "  train <project> [-p k=v ...] [--wait] [--follow]      (submit type=train)\n"
            "  build <project> [--targets train,eval] [--clean] [--arch 120] [--wait]\n"
            "  jobs [--status S] [--project P] [--type T] [--limit N]\n"
            "  job <id> | stop <id>\n"
            "  logs <id> [--follow] [--offset N] [--limit N]\n"
            "  metrics <id> [--keys ce,lr,grad_norm,tps] [--from N] [--to N] [--rank R] [--csv]\n"
            "  grad order <id> | grad <id> [--step N] [--rank R] [--csv]\n"
            "  profile <id> [--upload trace.json]\n"
            "  report <id> [--format md|json] [--out FILE]\n"
            "  artifacts <id> [name] [--out FILE]\n"
            "  docker ps | docker images\n"
            "  events [--topics job,metrics,grad,gpu] [--count N]\n"
            "global: --json (compact output)\n");
    }
} // namespace

int main(int argc, char **argv)
{
    Server srv;
    if (const char *e = std::getenv("TC_SERVER"))
    {
        std::string u = e;
        if (u.rfind("http://", 0) == 0)
            u = u.substr(7);
        auto colon = u.find(':');
        if (colon == std::string::npos)
            srv.host = u;
        else
        {
            srv.host = u.substr(0, colon);
            srv.port = u.substr(colon + 1);
        }
    }
    if (const char *e = std::getenv("TC_TOKEN"))
        srv.token = e;

    int i = 1;
    for (; i < argc; ++i)
    {
        std::string s = argv[i];
        if (s == "--server" && i + 1 < argc)
        {
            std::string u = argv[++i];
            if (u.rfind("http://", 0) == 0)
                u = u.substr(7);
            auto colon = u.find(':');
            if (colon == std::string::npos)
                srv.host = u;
            else
            {
                srv.host = u.substr(0, colon);
                srv.port = u.substr(colon + 1);
            }
        }
        else if (s == "--token" && i + 1 < argc)
            srv.token = argv[++i];
        else if (s == "--json")
            srv.compact = true;
        else
            break;
    }
    if (i >= argc)
    {
        usage();
        return 2;
    }
    std::string cmd = argv[i++];
    Args a = parse_args(argc, argv, i);

    try
    {
        if (cmd == "health")
            print_j(srv, call(srv, "GET", "/api/health"));
        else if (cmd == "system")
            print_j(srv, call(srv, "GET", "/api/system"));
        else if (cmd == "gpus")
            print_j(srv, call(srv, "GET", "/api/gpus"));
        else if (cmd == "projects")
        {
            std::string sub = a.pos.empty() ? "list" : a.pos[0];
            if (sub == "list")
                print_j(srv, call(srv, "GET", "/api/projects"));
            else if (sub == "add")
            {
                J body = {{"root", a.get("root")},
                          {"name", a.get("name", a.get("root"))},
                          {"kind", a.get("kind", "metamodel")}};
                if (a.has("id"))
                    body["id"] = a.get("id");
                if (a.get("root").empty())
                    die("projects add requires --root");
                print_j(srv, call(srv, "POST", "/api/projects", body.dump()));
            }
            else if (sub == "rm" && a.pos.size() > 1)
                print_j(srv, call(srv, "DELETE", "/api/projects/" + a.pos[1]));
            else
                die("usage: tc projects list|add|rm");
        }
        else if (cmd == "schema")
        {
            if (a.pos.empty())
                die("usage: tc schema <project> [--type T]");
            std::string path = "/api/projects/" + a.pos[0] + "/schema";
            if (a.has("type"))
                path += "?type=" + q(a.get("type"));
            print_j(srv, call(srv, "GET", path));
        }
        else if (cmd == "submit")
        {
            if (a.pos.size() < 2)
                die("usage: tc submit <project> <type> [-p k=v ...]");
            std::string type = a.pos[1];
            a.pos.erase(a.pos.begin() + 1);
            return do_submit(srv, a, type);
        }
        else if (cmd == "train")
            return do_submit(srv, a, "train");
        else if (cmd == "build")
        {
            if (a.pos.empty())
                die("usage: tc build <project> [--targets train,eval] [--clean] [--arch 120]");
            J params = collect_params(a);
            if (a.has("targets"))
                params["targets"] = a.get("targets");
            if (a.has("clean"))
                params["clean"] = true;
            if (a.has("arch"))
                params["cuda_arch"] = a.get("arch");
            if (a.has("type"))
                params["build_type"] = a.get("type");
            J body = {{"project_id", a.pos[0]}, {"type", "build"}, {"params", params}};
            J rec = call(srv, "POST", "/api/jobs", body.dump());
            std::string id = rec.at("id").get<std::string>();
            if (a.has("follow"))
                follow_logs(srv, id);
            if (a.has("wait") || a.has("follow"))
                return wait_job(srv, id);
            print_j(srv, rec);
        }
        else if (cmd == "jobs")
        {
            std::string path = "/api/jobs?limit=" + a.get("limit", "50");
            if (a.has("status"))
                path += "&status=" + q(a.get("status"));
            if (a.has("project"))
                path += "&project=" + q(a.get("project"));
            if (a.has("type"))
                path += "&type=" + q(a.get("type"));
            print_j(srv, call(srv, "GET", path));
        }
        else if (cmd == "job")
        {
            if (a.pos.empty())
                die("usage: tc job <id>");
            print_j(srv, call(srv, "GET", "/api/jobs/" + a.pos[0]));
        }
        else if (cmd == "stop")
        {
            if (a.pos.empty())
                die("usage: tc stop <id>");
            print_j(srv, call(srv, "POST", "/api/jobs/" + a.pos[0] + "/stop"));
        }
        else if (cmd == "logs")
        {
            if (a.pos.empty())
                die("usage: tc logs <id> [--follow]");
            std::string id = a.pos[0];
            if (a.has("follow"))
            {
                follow_logs(srv, id);
                return 0;
            }
            J j = call(srv, "GET", "/api/jobs/" + id + "/logs?offset=" + a.get("offset", "0") +
                                       "&limit=" + a.get("limit", "500"));
            for (auto &l : j["lines"])
                std::printf("%s\n", l.value("line", "").c_str());
            if (!srv.compact)
                std::fprintf(stderr, "[tc] %zu/%zu lines (next_offset=%zu)\n",
                             (size_t)j["lines"].size(), (size_t)j.value("total", (int64_t)0),
                             (size_t)j.value("next_offset", (int64_t)0));
        }
        else if (cmd == "metrics")
        {
            if (a.pos.empty())
                die("usage: tc metrics <id> [--keys ce,lr] [--csv]");
            std::string path = "/api/jobs/" + a.pos[0] + "/metrics?limit=" + a.get("limit", "5000");
            if (a.has("keys"))
                path += "&keys=" + q(a.get("keys"));
            if (a.has("from"))
                path += "&from=" + a.get("from");
            if (a.has("to"))
                path += "&to=" + a.get("to");
            if (a.has("rank"))
                path += "&rank=" + a.get("rank");
            J rows = call(srv, "GET", path);
            if (a.has("csv"))
                csv_metrics(rows);
            else
                print_j(srv, rows);
        }
        else if (cmd == "grad")
        {
            if (a.pos.empty())
                die("usage: tc grad order <id> | tc grad <id> [--step N] [--csv]");
            if (a.pos[0] == "order")
            {
                if (a.pos.size() < 2)
                    die("usage: tc grad order <id>");
                print_j(srv, call(srv, "GET", "/api/jobs/" + a.pos[1] + "/grad/order"));
            }
            else
            {
                std::string path = "/api/jobs/" + a.pos[0] + "/grad?limit=" + a.get("limit", "200");
                if (a.has("step"))
                    path += "&step=" + a.get("step");
                if (a.has("rank"))
                    path += "&rank=" + a.get("rank");
                J rows = call(srv, "GET", path);
                if (a.has("csv"))
                {
                    std::cout << "step,rank,name,grad_l2\n";
                    for (auto &r : rows)
                        for (auto &n : r["norms"])
                            std::cout << r.value("step", (int64_t)0) << ","
                                      << r.value("rank", (int64_t)0) << ","
                                      << n[0].get<std::string>() << "," << n[1].dump() << "\n";
                }
                else
                    print_j(srv, rows);
            }
        }
        else if (cmd == "profile")
        {
            if (a.pos.empty())
                die("usage: tc profile <id> [--upload trace.json]");
            if (a.has("upload"))
            {
                std::string data = read_all(a.get("upload"));
                print_j(srv, call(srv, "POST", "/api/jobs/" + a.pos[0] + "/profile", data));
            }
            else
                print_j(srv, call(srv, "GET", "/api/jobs/" + a.pos[0] + "/profile"));
        }
        else if (cmd == "report")
        {
            if (a.pos.empty())
                die("usage: tc report <id> [--format md|json] [--out FILE]");
            std::string fmt = a.get("format", "md");
            auto r = http(srv, "GET", "/api/jobs/" + a.pos[0] + "/report?format=" + fmt);
            if (r.status != 200)
                die("report: HTTP " + std::to_string(r.status) + ": " + r.body);
            if (a.has("out"))
            {
                std::ofstream f(a.get("out"), std::ios::binary);
                f << r.body;
                std::fprintf(stderr, "[tc] saved %s (%zu bytes)\n", a.get("out").c_str(), r.body.size());
            }
            else
                std::cout << r.body << "\n";
        }
        else if (cmd == "artifacts")
        {
            if (a.pos.empty())
                die("usage: tc artifacts <id> [name] [--out FILE]");
            if (a.pos.size() < 2)
            {
                print_j(srv, call(srv, "GET", "/api/jobs/" + a.pos[0] + "/artifacts"));
            }
            else
            {
                auto r = http(srv, "GET",
                              "/api/jobs/" + a.pos[0] + "/artifacts/" + q(a.pos[1]));
                if (r.status != 200)
                    die("artifact: HTTP " + std::to_string(r.status));
                if (a.has("out"))
                {
                    std::ofstream f(a.get("out"), std::ios::binary);
                    f << r.body;
                    std::fprintf(stderr, "[tc] saved %s (%zu bytes)\n", a.get("out").c_str(), r.body.size());
                }
                else
                    std::cout << r.body;
            }
        }
        else if (cmd == "docker")
        {
            std::string sub = a.pos.empty() ? "ps" : a.pos[0];
            if (sub != "ps" && sub != "images")
                die("usage: tc docker ps|images");
            print_j(srv, call(srv, "GET", "/api/docker/" + sub));
        }
        else if (cmd == "events")
        {
            int count = std::atoi(a.get("count", "0").c_str());
            int seen = 0;
            sse(srv, a.get("topics"), [&](const std::string &ev, const J &d) {
                std::cout << "event: " << ev << "\ndata: " << (srv.compact ? d.dump() : d.dump()) << "\n\n";
                std::cout.flush();
                return count <= 0 || ++seen < count;
            });
        }
        else if (cmd == "help" || cmd == "--help" || cmd == "-h")
            usage();
        else
        {
            std::fprintf(stderr, "tc: unknown command: %s\n", cmd.c_str());
            usage();
            return 2;
        }
    }
    catch (const std::exception &e)
    {
        die(e.what());
    }
    return 0;
}