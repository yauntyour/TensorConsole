#pragma once
#include "common.hpp"

#include <deque>

namespace tc
{
    // File-backed registry: projects.json + jobs/<id>/job.json + JSONL series.
    class Store
    {
    public:
        explicit Store(fs::path data_dir);

        fs::path job_dir(const std::string &id) const;

        J list_projects();
        J get_project(const std::string &id);
        J add_project(J proj);
        bool remove_project(const std::string &id);
        J update_project(const std::string &id, const J &patch);

        J create_job(J rec);
        J get_job(const std::string &id);
        std::vector<J> list_jobs(const std::string &status = "", const std::string &project = "",
                                 const std::string &type = "", size_t limit = 200);
        void update_job(const std::string &id, const J &patch);
        bool remove_job(const std::string &id);

        void append_jsonl(const std::string &id, const std::string &file, const J &row);
        std::vector<J> read_jsonl(const std::string &id, const std::string &file);
        void append_log(const std::string &id, const std::string &line);

    private:
        void save_projects();
        fs::path dir_;
        std::mutex mx_;
        J projects_ = J::array();
        std::map<std::string, J> jobs_;
    };
}