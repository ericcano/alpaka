/**
 * 
 */

 #pragma once

#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <functional>
#include <future>
#include <mutex>
#include <utility>

namespace alpaka::cuda::detail {

class SingleThread {
public:
    // Post a callable to the single-threaded CUDA TBB arena
    template<typename F>
    static void post(F&& f) {
        auto& instance = getInstance();
        instance.m_arena.enqueue([&instance, f](){
            instance.m_group.run(std::move(f));
        });
    }

    template <typename F, typename R>
    static R postAndReturn(F&& f) {
        std::promise<R> promise;
        auto future = promise.get_future();

        post([&promise, func = std::forward<F>(f)]() mutable {
            try {
                if constexpr (std::is_void_v<R>) {
                    func();
                    promise.set_value();
                } else {
                    promise.set_value(func());
                }
            } catch (...) {
                promise.set_exception(std::current_exception());
            }
        });

        return future.get();
    }

    ~SingleThread() {
        m_group.wait();
    } 

private:
    SingleThread() : m_arena(1) {}

    static SingleThread& getInstance() {
        static SingleThread instance;
        return instance;
    }

    tbb::task_arena m_arena;
    tbb::task_group m_group;
};

} // namespace alpaka::cuda::detail