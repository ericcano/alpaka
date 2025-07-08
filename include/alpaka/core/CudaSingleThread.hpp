/**
 * 
 */

 #pragma once

#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <functional>
#include <mutex>

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