#include "events.hpp"

#include <algorithm>

namespace tc
{
    std::shared_ptr<SseConn> EventBus::subscribe(const std::string &topics,
                                                 std::function<void(const std::string &)> write)
    {
        auto c = std::make_shared<SseConn>();
        c->topics = topics;
        c->write = std::move(write);
        std::lock_guard<std::mutex> lk(mx_);
        if (conns_.size() >= kMaxConns)
            return nullptr;
        conns_.push_back(c);
        return c;
    }

    void EventBus::unsubscribe(const std::shared_ptr<SseConn> &c)
    {
        {
            std::lock_guard<std::mutex> lk(c->mx);
            c->alive = false;
        }
        c->cv.notify_all();
        std::lock_guard<std::mutex> lk(mx_);
        conns_.erase(std::remove(conns_.begin(), conns_.end(), c), conns_.end());
    }

    void EventBus::close_all()
    {
        std::vector<std::shared_ptr<SseConn>> all;
        {
            std::lock_guard<std::mutex> lk(mx_);
            all.swap(conns_);
        }
        for (auto &c : all)
        {
            {
                std::lock_guard<std::mutex> lk(c->mx);
                c->alive = false;
            }
            c->cv.notify_all();
        }
    }

    bool EventBus::wants(const std::shared_ptr<SseConn> &c, const std::string &event)
    {
        if (c->topics.empty() || event == "heartbeat")
            return true;
        // topic = first segment of event name (job.status -> job)
        auto dot = event.find('.');
        std::string topic = dot == std::string::npos ? event : event.substr(0, dot);
        for (auto &t : split(c->topics, ','))
            if (trim(t) == topic || trim(t) == event)
                return true;
        return false;
    }

    void EventBus::publish(const std::string &event, const J &data)
    {
        std::string frame = "event: " + event + "\ndata: " + data.dump() + "\n\n";
        std::vector<std::shared_ptr<SseConn>> targets;
        {
            std::lock_guard<std::mutex> lk(mx_);
            for (auto &c : conns_)
                if (c->alive && wants(c, event))
                    targets.push_back(c);
        }
        for (auto &c : targets)
        {
            std::lock_guard<std::mutex> lk(c->mx);
            if (!c->alive)
                continue;
            try
            {
                c->write(frame);
            }
            catch (...)
            {
                c->alive = false;
                c->cv.notify_all();
            }
        }
    }

    size_t EventBus::connection_count()
    {
        std::lock_guard<std::mutex> lk(mx_);
        return conns_.size();
    }
}