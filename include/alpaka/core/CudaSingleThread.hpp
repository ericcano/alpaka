/**
 * 
 */

 #pragma once

#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <utility>
#include <thread>
#include <unistd.h>

namespace alpaka::cuda::detail {

class SingleThread {
private:
    // Post a callable to the single-threaded CUDA TBB arena
    template<typename F>
    static void post(F&& f) {
        auto& instance = getInstance();
        // Use shared_ptr to ensure copyability
        auto f_ptr = std::make_shared<std::decay_t<F>>(std::forward<F>(f));
        instance.m_arena.enqueue([&instance, f_ptr]() {
            instance.m_group.run([f_ptr]() { (*f_ptr)(); });
        });
    }

public:
    template <typename R, typename F>
    static R postAndReturn(F&& f) {
        std::promise<R> promise;
        auto future = promise.get_future();

        post([promise_ptr = &promise, func = std::forward<F>(f)]() mutable {
            std::cout << "Lambda START [" << promise_ptr << "] (thread 0x" << std::hex << std::this_thread::get_id() 
                 << "/"  << std::dec << gettid() << ")" << std::endl;
            try {
                if constexpr (std::is_void_v<R>) {
                    func();
                    promise_ptr->set_value();
                } else {
                    promise_ptr->set_value(func());
                }
            } catch (...) {
                promise_ptr->set_exception(std::current_exception());
            }
            std::cout << "Lambda END [" << promise_ptr << "] (thread 0x" << std::hex << std::this_thread::get_id() 
                 << "/"  << std::dec << gettid() << ")" << std::endl;
        });

        std::cout << "Waiting on promise [" << &promise << "] (thread 0x" << std::hex << std::this_thread::get_id() 
                 << "/"  << std::dec << gettid() << ")" << std::endl;
        if constexpr (std::is_void_v<R>) {
            future.wait();
            std::cout << "Promise fulfilled [" << &promise << "] (thread 0x" << std::hex << std::this_thread::get_id() 
                     << "/"  << std::dec << gettid() << ")" << std::endl;
            return;
        } else { 
           R ret = future.get();
           std::cout << "Promise fulfilled [" << &promise << "] (thread 0x" << std::hex << std::this_thread::get_id() 
                     << "/"  << std::dec << gettid() << ")" << std::endl;
           return ret;
        }
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