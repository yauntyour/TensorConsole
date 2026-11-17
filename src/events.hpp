#pragma once
#include "common.hpp"

#include <condition_variable>
#include <memory>

namespace tc
{
    // One SSE connection. publish() writes under mx; the HTTP handler thread
    // blocks on cv until the connection dies (write error or unsubscribe).
    struct SseConn
    {
        std::mutex mx;
        std::condition_variable cv;
        std::function<void(const std::string &)> write;
        bool alive = true;
        std::string topics; // comma list, empty = all
    };

    class EventBus
    {
    public:
        static constexpr size_t kMaxConns = 32;

        std::shared_ptr<SseConn> subscribe(const std::string &topics,
                                           std::function<void(const std::string &)> write);
        void unsubscribe(const std::shared_ptr<SseConn> &c);
        void close_all();
        void publish(const std::string &event, const J &data);
        size_t connection_count();

    private:
        bool wants(const std::shared_ptr<SseConn> &c, const std::string &event);
        std::mutex mx_;
        std::vector<std::shared_ptr<SseConn>> conns_;
    };
}