#include "api.hpp"

#include "adapter.hpp"
#include "events.hpp"
#include "jobs.hpp"
#include "monitor.hpp"
#include "store.hpp"

#include <servic.hpp>

#include <boost/asio.hpp>

#include <algorithm>
#include <csignal>
#include <set>

namespace asio = boost::asio;

namespace tc
{
    namespace
    {
        struct Ctx
        {
            Config &cfg;
            Store &store;
            JobManager &jobs;
            Monitor &mon;
            EventBus &bus;
            int64_t started = now_ms();
        };
        using CtxPtr = std::shared_ptr<Ctx>;

        bool authorized(const Request &req, const Config &cfg)
        {
            if (cfg.token.empty())
                return true;
            if (req.header("Authorization") == "Bearer " + cfg.token)
                return true;
            return req.param("token") == cfg.token;
        }

        int64_t qint(const Request &req, const std::string &key, int64_t def)
        {
            auto s = req.param(key);
            if (s.empty())
                return def;
            try
            {
                return std::stoll(s);
            }
            catch (...)
            {
                return def;
            }
        }

        using ApiFn = std::function<std::string(const Request &, const std::map<std::string, std::string> &)>;

        rt::HandlerFunc multi(CtxPtr ctx, std::map<std::string, ApiFn> fns, bool auth = true);

        rt::HandlerFunc wrap(CtxPtr ctx, const std::string &method, ApiFn fn, bool auth = true)
        {
            return multi(ctx, {{method, std::move(fn)}}, auth);
        }

        // rt::router keeps ONE handler per path, so GET/POST/DELETE on the same
        // path must be merged into a single method-dispatching handler.
        rt::HandlerFunc multi(CtxPtr ctx, std::map<std::string, ApiFn> fns, bool auth)
        {
            return [ctx, fns, auth](std::string &raw, std::string &out,
                                    const std::map<std::string, std::string> &params) -> int {
                try
                {
                    Request req = parse_request(raw);
                    if (req.method == "OPTIONS")
                    {
                        out = http_response(204, "text/plain", "");
                        return rt::FLAG_DONE;
                    }
                    auto it = fns.find(req.method);
                    if (it == fns.end())
                    {
                        out = error_resp(405, "method not allowed: " + req.method);
                        return rt::FLAG_DONE;
                    }
                    if (auth && !authorized(req, ctx->cfg))
                    {
                        out = error_resp(401, "unauthorized: missing or bad token");
                        return rt::FLAG_DONE;
                    }
                    out = it->second(req, params);
                }
                catch (const std::exception &e)
                {
                    out = error_resp(400, e.what());
                }
                return rt::FLAG_DONE;
            };
        }

        J require_object(const Request &req)
        {
            J body = req.json_body();
            if (!body.is_object())
                throw std::runtime_error("request body must be a JSON object");
            return body;
        }

        std::string at(const std::map<std::string, std::string> &params, const std::string &key)
        {
            auto it = params.find(key);
            if (it == params.end())
                throw std::runtime_error("missing route param: " + key);
            return it->second;
        }

        // Reject traversal in user-supplied relative file names.
        bool safe_rel(const std::string &s)
        {
            if (s.empty() || s.find("..") != std::string::npos)
                return false;
            if (s[0] == '/' || s[0] == '\\' || s.find(':') != std::string::npos)
                return false;
            return true;
        }

        std::string serve_static(const fs::path &web, const std::string &rel)
        {
            if (!safe_rel(rel))
                return error_resp(400, "bad path");
            fs::path p = web / rel;
            std::error_code ec;
            if (!fs::is_regular_file(p, ec))
                return error_resp(404, "not found: " + rel);
            return file_resp(p);
        }

        J docker_query(const std::vector<std::string> &argv)
        {
            auto cap = run_capture("docker", argv, 10000);
            if (!cap.started)
                return J{{"available", false}, {"error", "docker CLI not available"}};
            J rows = J::array();
            for (auto &line : split(cap.output, '\n'))
            {
                auto t = trim(line);
                if (t.empty())
                    continue;
                auto j = J::parse(t, nullptr, false);
                if (j.is_object())
                    rows.push_back(j);
            }
            return J{{"available", true}, {"exit_code", cap.exit_code}, {"rows", rows}};
        }
    } // namespace

    static void register_api(rt::router &ros, Config &cfg, Store &store, JobManager &jobs,
                             Monitor &mon, EventBus &bus)
    {
        auto ctx = std::make_shared<Ctx>(Ctx{cfg, store, jobs, mon, bus});

        // ---------------------------------------------------------- meta
        ros.on("/api/health", wrap(ctx, "GET", [ctx](const Request &, auto &) {
                   return json_resp(J{{"ok", true},
                                      {"name", "TensorConsole"},
                                      {"version", "1.0.0"},
                                      {"uptime_ms", now_ms() - ctx->started},
                                      {"sse_connections", ctx->bus.connection_count()}});
               }, false));

        ros.on("/api/system", wrap(ctx, "GET", [ctx](const Request &, auto &) {
                   J j = ctx->mon.system_info();
                   j["bind"] = ctx->cfg.bind;
                   j["port"] = ctx->cfg.port;
                   j["auth"] = !ctx->cfg.token.empty();
                   j["data_dir"] = fs::absolute(ctx->cfg.data_dir).string();
                   return json_resp(j);
               }));

        ros.on("/api/gpus", wrap(ctx, "GET", [ctx](const Request &req, auto &) {
                   J j = ctx->mon.gpus();
                   if (req.param("history", "1") != "0")
                       j["history"] = ctx->mon.gpu_history();
                   return json_resp(j);
               }));

        // ---------------------------------------------------------- projects
        ros.on("/api/projects", multi(ctx, {
            {"GET", [ctx](const Request &, auto &) {
                 return json_resp(ctx->store.list_projects());
             }},
            {"POST", [ctx](const Request &req, auto &) {
                 J body = require_object(req);
                 if (!body.contains("root") || !fs::exists(body["root"].get<std::string>()))
                     throw std::runtime_error("project root does not exist");
                 if (!body.contains("id") || body["id"].get<std::string>().empty())
                     body["id"] = new_id("p");
                 body["created"] = iso_utc();
                 J proj = ctx->store.add_project(body);
                 return json_resp(proj, 201);
             }},
        }));

        ros.on("/api/projects/:id", multi(ctx, {
            {"GET", [ctx](const Request &, auto &p) {
                 J proj = ctx->store.get_project(at(p, "id"));
                 if (proj.is_null())
                     return error_resp(404, "unknown project");
                 return json_resp(proj);
             }},
            {"DELETE", [ctx](const Request &, auto &p) {
                 if (!ctx->store.remove_project(at(p, "id")))
                     return error_resp(404, "unknown project");
                 return json_resp(J{{"ok", true}});
             }},
            {"PUT", [ctx](const Request &req, auto &p) {
                 J body = require_object(req);
                 if (body.contains("root") && !fs::exists(body["root"].get<std::string>()))
                     throw std::runtime_error("project root does not exist");
                 J proj = ctx->store.update_project(at(p, "id"), body);
                 if (proj.is_null())
                     return error_resp(404, "unknown project");
                 return json_resp(proj);
             }},
        }));

        ros.on("/api/projects/:id/schema", wrap(ctx, "GET", [ctx](const Request &req, auto &p) {
                   J proj = ctx->store.get_project(at(p, "id"));
                   if (proj.is_null())
                       return error_resp(404, "unknown project");
                   J schema = adapter::schema_for(proj.value("kind", "generic"));
                   std::string type = req.param("type");
                   if (!type.empty())
                   {
                       if (!schema.contains(type))
                           return error_resp(404, "unknown job type: " + type);
                       schema = schema[type];
                   }
                   return json_resp(J{{"project", proj}, {"schema", schema}});
               }));

        // ---------------------------------------------------------- jobs
        ros.on("/api/jobs", multi(ctx, {
            {"GET", [ctx](const Request &req, auto &) {
                 auto rows = ctx->store.list_jobs(req.param("status"), req.param("project"),
                                                  req.param("type"), (size_t)qint(req, "limit", 200));
                 return json_resp(rows);
             }},
            {"POST", [ctx](const Request &req, auto &) {
                 J rec = ctx->jobs.submit(require_object(req));
                 return json_resp(rec, 201);
             }},
        }));

        ros.on("/api/jobs/:id", multi(ctx, {
            {"GET", [ctx](const Request &, auto &p) {
                 J job = ctx->store.get_job(at(p, "id"));
                 if (job.is_null())
                     return error_resp(404, "unknown job");
                 return json_resp(job);
             }},
            {"DELETE", [ctx](const Request &, auto &p) {
                 std::string id = at(p, "id");
                 J job = ctx->store.get_job(id);
                 if (job.is_null())
                     return error_resp(404, "unknown job");
                 if (job.value("status", "") == "running")
                     return error_resp(409, "job is running; stop it first");
                 ctx->store.remove_job(id);
                 return json_resp(J{{"ok", true}});
             }},
            {"PATCH", [ctx](const Request &req, auto &p) {
                 std::string id = at(p, "id");
                 if (ctx->store.get_job(id).is_null())
                     return error_resp(404, "unknown job");
                 J body = require_object(req);
                 J patch = J::object();
                 for (auto &k : {"title", "notes"})
                     if (body.contains(k) && body[k].is_string())
                         patch[k] = body[k];
                 ctx->store.update_job(id, patch);
                 return json_resp(ctx->store.get_job(id));
             }},
        }));

        ros.on("/api/jobs/:id/stop", wrap(ctx, "POST", [ctx](const Request &, auto &p) {
                   return json_resp(ctx->jobs.stop(at(p, "id")));
               }));

        ros.on("/api/jobs/:id/logs", wrap(ctx, "GET", [ctx](const Request &req, auto &p) {
                   return json_resp(ctx->jobs.logs(at(p, "id"), (size_t)qint(req, "offset", 0),
                                                   (size_t)qint(req, "limit", 500)));
               }));

        ros.on("/api/jobs/:id/metrics", wrap(ctx, "GET", [ctx](const Request &req, auto &p) {
                   auto rows = ctx->jobs.metrics(at(p, "id"), req.param("keys"), qint(req, "from", -1),
                                                 qint(req, "to", -1), qint(req, "rank", -1),
                                                 (size_t)qint(req, "limit", 5000));
                   return json_resp(rows);
               }));

        ros.on("/api/jobs/:id/grad/order", wrap(ctx, "GET", [ctx](const Request &, auto &p) {
                   return json_resp(ctx->jobs.grad_order(at(p, "id")));
               }));

        ros.on("/api/jobs/:id/grad", wrap(ctx, "GET", [ctx](const Request &req, auto &p) {
                   auto rows = ctx->jobs.grad_steps(at(p, "id"), qint(req, "step", -1),
                                                    qint(req, "from", -1), qint(req, "to", -1),
                                                    qint(req, "rank", -1), (size_t)qint(req, "limit", 200));
                   return json_resp(rows);
               }));

        ros.on("/api/jobs/:id/profile", multi(ctx, {
            {"GET", [ctx](const Request &, auto &p) {
                 return json_resp(ctx->jobs.profile(at(p, "id")));
             }},
            {"POST", [ctx](const Request &req, auto &p) {
                 J body = req.json_body();
                 J trace = body.is_array() ? J{{"traceEvents", body}} : body;
                 if (!trace.contains("traceEvents"))
                     throw std::runtime_error("expected Chrome Trace JSON with traceEvents");
                 ctx->jobs.upload_profile(at(p, "id"), trace);
                 return json_resp(J{{"ok", true}});
             }},
        }));

        ros.on("/api/jobs/:id/report", wrap(ctx, "GET", [ctx](const Request &req, auto &p) {
                   std::string fmt = req.param("format", "md");
                   J rep = ctx->jobs.report(at(p, "id"), fmt);
                   if (fmt == "json")
                       return json_resp(rep);
                   return http_response(200, "text/markdown; charset=utf-8", rep["md"].get<std::string>());
               }));

        ros.on("/api/jobs/:id/artifacts", wrap(ctx, "GET", [ctx](const Request &, auto &p) {
                   fs::path jd = ctx->store.job_dir(at(p, "id"));
                   J out = J::array();
                   auto add = [&](const fs::path &f, const std::string &prefix) {
                       std::error_code ec;
                       auto sz = fs::file_size(f, ec);
                       out.push_back(J{{"name", prefix + f.filename().string()},
                                       {"size", ec ? 0 : (int64_t)sz}});
                   };
                   for (auto &f : list_dir(jd))
                       if (fs::is_regular_file(f))
                           add(f, "");
                   for (auto &f : list_dir(jd / "artifacts"))
                       if (fs::is_regular_file(f))
                           add(f, "artifacts/");
                   return json_resp(out);
               }));

        ros.on("/api/jobs/:id/artifacts/:name", wrap(ctx, "GET", [ctx](const Request &req, auto &p) {
                   std::string id = at(p, "id");
                   std::string rel = req.param("path", at(p, "name"));
                   if (!safe_rel(rel))
                       return error_resp(400, "bad artifact path");
                   fs::path jd = ctx->store.job_dir(id);
                   fs::path f = jd / rel;
                   if (!fs::exists(f))
                       f = jd / "artifacts" / rel;
                   return file_resp(f);
               }));

        // ---------------------------------------------------------- docker / deploy
        ros.on("/api/docker/ps", wrap(ctx, "GET", [](const Request &, auto &) {
                   return json_resp(docker_query({"ps", "-a", "--format", "{{json .}}"}));
               }));

        ros.on("/api/docker/images", wrap(ctx, "GET", [](const Request &, auto &) {
                   return json_resp(docker_query({"images", "--format", "{{json .}}"}));
               }));

        ros.on("/api/docker/action", wrap(ctx, "POST", [](const Request &req, auto &) {
                   J body = require_object(req);
                   std::string action = body.value("action", "");
                   std::string id = body.value("id", "");
                   static const std::set<std::string> allowed = {"start", "stop", "restart", "rm", "rmi"};
                   if (!allowed.count(action))
                       throw std::runtime_error("action must be one of start/stop/restart/rm/rmi");
                   if (id.empty() || id.find_first_of(" \t\r\n\"'&|;") != std::string::npos)
                       throw std::runtime_error("invalid container/image id");
                   auto cap = run_capture("docker", {action, id}, 20000);
                   if (!cap.started)
                       return error_resp(503, "docker CLI not available");
                   return json_resp(J{{"ok", cap.exit_code == 0},
                                      {"exit_code", cap.exit_code},
                                      {"output", trim(cap.output)}});
               }));

        ros.on("/api/deploy", wrap(ctx, "POST", [ctx](const Request &req, auto &) {
                   J body = require_object(req);
                   body["type"] = "deploy";
                   return json_resp(ctx->jobs.submit(body), 201);
               }));

        // ---------------------------------------------------------- SSE
        ros.on_stream("/api/events", [ctx](std::string &raw, rt::WriteCallback write,
                                           const std::map<std::string, std::string> &) {
            Request req = parse_request(raw);
            if (!authorized(req, ctx->cfg))
            {
                write(http_response(401, "text/plain", "unauthorized"));
                return;
            }
            write("HTTP/1.1 200 OK\r\n"
                  "Content-Type: text/event-stream\r\n"
                  "Cache-Control: no-cache\r\n"
                  "Connection: close\r\n"
                  "Access-Control-Allow-Origin: *\r\n\r\n");
            write("retry: 3000\n\n");
            write("event: hello\ndata: " + J{{"ok", true}, {"t", now_ms()}}.dump() + "\n\n");
            auto conn = ctx->bus.subscribe(req.param("topics"), write);
            if (!conn)
            {
                write("event: error\ndata: {\"msg\":\"too many SSE connections\"}\n\n");
                return;
            }
            {
                std::unique_lock<std::mutex> lk(conn->mx);
                conn->cv.wait(lk, [&conn] { return !conn->alive; });
            }
            ctx->bus.unsubscribe(conn);
        });

        // ---------------------------------------------------------- static web UI
        ros.on("/", wrap(ctx, "GET", [ctx](const Request &, auto &) {
                   return serve_static(ctx->cfg.web_dir, "index.html");
               }, false));
        ros.on("/index.html", wrap(ctx, "GET", [ctx](const Request &, auto &) {
                   return serve_static(ctx->cfg.web_dir, "index.html");
               }, false));
        ros.on("/css/:f", wrap(ctx, "GET", [ctx](const Request &, auto &p) {
                   return serve_static(ctx->cfg.web_dir, "css/" + at(p, "f"));
               }, false));
        ros.on("/js/:f", wrap(ctx, "GET", [ctx](const Request &, auto &p) {
                   return serve_static(ctx->cfg.web_dir, "js/" + at(p, "f"));
               }, false));
        ros.on("/vendor/:d/:f", wrap(ctx, "GET", [ctx](const Request &, auto &p) {
                   return serve_static(ctx->cfg.web_dir, "vendor/" + at(p, "d") + "/" + at(p, "f"));
               }, false));
        ros.on("/favicon.ico", wrap(ctx, "GET", [](const Request &, auto &) {
                   return http_response(204, "image/x-icon", "");
               }, false));
    }

    void run_server(Config &cfg, Store &store, JobManager &jobs, Monitor &mon, EventBus &bus)
    {
        rt::router ros;
        register_api(ros, cfg, store, jobs, mon, bus);

        asio::io_context io;
        // 64 MiB request cap: Chrome-trace profile uploads can be large.
        servic::Server srv(io, cfg.bind, cfg.port, 64 * 1024 * 1024);
        srv.start(ros);
        mon.start();

        asio::signal_set signals(io, SIGINT, SIGTERM);
        signals.async_wait([&](const boost::system::error_code &, int sig) {
            std::printf("\n[tcd] signal %d — shutting down\n", sig);
            bus.close_all();
            srv.stop_handlers(5000);
            io.stop();
        });

        std::printf("[tcd] TensorConsole 1.0.0  http://%s:%u  (data: %s)\n",
                    cfg.bind.c_str(), (unsigned)srv.local_port(),
                    fs::absolute(cfg.data_dir).string().c_str());
        std::fflush(stdout);
        io.run();
        mon.stop();
        std::printf("[tcd] bye\n");
    }
}