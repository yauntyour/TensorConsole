#include "store.hpp"

#include <algorithm>
#include <fstream>

namespace tc
{
    Store::Store(fs::path data_dir) : dir_(std::move(data_dir))
    {
        ensure_dir(dir_);
        ensure_dir(dir_ / "jobs");
        auto pf = dir_ / "projects.json";
        if (fs::exists(pf))
        {
            auto j = J::parse(read_file(pf), nullptr, false);
            if (j.is_array())
                projects_ = j;
        }
        for (auto &d : list_dir(dir_ / "jobs"))
        {
            auto jf = d / "job.json";
            if (!fs::exists(jf))
                continue;
            auto j = J::parse(read_file(jf), nullptr, false);
            if (j.is_object() && j.contains("id"))
                jobs_[j["id"].get<std::string>()] = j;
        }
    }

    fs::path Store::job_dir(const std::string &id) const { return dir_ / "jobs" / id; }

    void Store::save_projects() { write_file_atomic(dir_ / "projects.json", projects_.dump(2)); }

    J Store::list_projects()
    {
        std::lock_guard<std::mutex> lk(mx_);
        return projects_;
    }

    J Store::get_project(const std::string &id)
    {
        std::lock_guard<std::mutex> lk(mx_);
        for (auto &p : projects_)
            if (p.value("id", "") == id)
                return p;
        return J();
    }

    J Store::add_project(J proj)
    {
        std::lock_guard<std::mutex> lk(mx_);
        for (auto &p : projects_)
            if (p.value("id", "") == proj.value("id", ""))
                throw std::runtime_error("project id already exists: " + proj.value("id", ""));
        projects_.push_back(proj);
        save_projects();
        return proj;
    }

    bool Store::remove_project(const std::string &id)
    {
        std::lock_guard<std::mutex> lk(mx_);
        for (auto it = projects_.begin(); it != projects_.end(); ++it)
            if (it->value("id", "") == id)
            {
                projects_.erase(it);
                save_projects();
                return true;
            }
        return false;
    }

    J Store::update_project(const std::string &id, const J &patch)
    {
        std::lock_guard<std::mutex> lk(mx_);
        for (auto &p : projects_)
            if (p.value("id", "") == id)
            {
                for (auto &kv : patch.items())
                    if (kv.key() != "id" && kv.key() != "created")
                        p[kv.key()] = kv.value();
                save_projects();
                return p;
            }
        return J();
    }

    J Store::create_job(J rec)
    {
        std::string id = rec.at("id").get<std::string>();
        {
            std::lock_guard<std::mutex> lk(mx_);
            jobs_[id] = rec;
        }
        write_file_atomic(job_dir(id) / "job.json", rec.dump(2));
        return rec;
    }

    J Store::get_job(const std::string &id)
    {
        std::lock_guard<std::mutex> lk(mx_);
        auto it = jobs_.find(id);
        return it == jobs_.end() ? J() : it->second;
    }

    std::vector<J> Store::list_jobs(const std::string &status, const std::string &project,
                                    const std::string &type, size_t limit)
    {
        std::vector<J> out;
        std::lock_guard<std::mutex> lk(mx_);
        for (auto &kv : jobs_)
        {
            const J &j = kv.second;
            if (!status.empty() && j.value("status", "") != status)
                continue;
            if (!project.empty() && j.value("project_id", "") != project)
                continue;
            if (!type.empty() && j.value("type", "") != type)
                continue;
            out.push_back(j);
        }
        std::sort(out.begin(), out.end(), [](const J &a, const J &b) {
            return a.value("created", "") > b.value("created", "");
        });
        if (out.size() > limit)
            out.resize(limit);
        return out;
    }

    void Store::update_job(const std::string &id, const J &patch)
    {
        J rec;
        {
            std::lock_guard<std::mutex> lk(mx_);
            auto it = jobs_.find(id);
            if (it == jobs_.end())
                return;
            for (auto &kv : patch.items())
                it->second[kv.key()] = kv.value();
            rec = it->second;
        }
        write_file_atomic(job_dir(id) / "job.json", rec.dump(2));
    }

    bool Store::remove_job(const std::string &id)
    {
        {
            std::lock_guard<std::mutex> lk(mx_);
            if (!jobs_.erase(id))
                return false;
        }
        std::error_code ec;
        fs::remove_all(job_dir(id), ec);
        return true;
    }

    void Store::append_jsonl(const std::string &id, const std::string &file, const J &row)
    {
        append_file(job_dir(id) / file, row.dump() + "\n");
    }

    std::vector<J> Store::read_jsonl(const std::string &id, const std::string &file)
    {
        std::vector<J> out;
        std::ifstream f(job_dir(id) / file);
        std::string line;
        while (std::getline(f, line))
        {
            line = trim(line);
            if (line.empty())
                continue;
            auto j = J::parse(line, nullptr, false);
            if (j.is_object())
                out.push_back(std::move(j));
        }
        return out;
    }

    void Store::append_log(const std::string &id, const std::string &line)
    {
        append_file(job_dir(id) / "stdout.log", line + "\n");
    }
}