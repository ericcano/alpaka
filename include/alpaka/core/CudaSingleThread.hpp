/**
 * 
 */

 #pragma once

#include <tbb/concurrent_queue.h>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <atomic>
#include <utility>
#include <sys/syscall.h>
#include <unistd.h>
#include <sstream>
#include <iomanip>

namespace alpaka::cuda::detail {

// Dedicated worker for single-threaded task execution
class SingleThreadWorker {
public:
    struct Task {
        std::function<void()> func;
        bool terminate = false;
    };

    SingleThreadWorker() : m_running(true) {
        m_thread = std::thread([this] {
            while (m_running) {
                Task task;
                if (m_queue.try_pop(task)) {
                    if (task.terminate) break;
                    task.func();
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    ~SingleThreadWorker() {
        // Enqueue termination task
        m_queue.push(Task{[] {}, true});
        m_running = false;
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }

    void post(Task&& task) {
        m_queue.push(std::move(task));
    }

private:
    tbb::concurrent_queue<Task> m_queue;
    std::thread m_thread;
    std::atomic<bool> m_running;
};

class SingleThread {
private:
    SingleThread() = default;
    ~SingleThread() = default;

    static SingleThread& getInstance() {
        static SingleThread instance;
        return instance;
    }

    SingleThreadWorker m_worker;

    template<typename F>
    static void post(F&& f) {
        auto f_ptr = std::make_shared<std::decay_t<F>>(std::forward<F>(f));
        getInstance().m_worker.post(SingleThreadWorker::Task{
            [f_ptr]() { (*f_ptr)(); },
            false
        });
    }

public:
    template <typename R, typename F>
    static R postAndReturn(F&& f) {
        std::promise<R> promise;
        auto future = promise.get_future();

        post([promise_ptr = &promise, func = std::forward<F>(f)]() mutable {
            // std::cout << "Lambda START [" << promise_ptr << "] (thread 0x" << std::hex << std::this_thread::get_id() 
            //      << "/"  << std::dec << gettid() << ")" << std::endl;
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
            // std::cout << "Lambda END [" << promise_ptr << "] (thread 0x" << std::hex << std::this_thread::get_id() 
            //      << "/"  << std::dec << gettid() << ")" << std::endl;
        });

        // std::cout << "Waiting on promise [" << &promise << "] (thread 0x" << std::hex << std::this_thread::get_id() 
        //          << "/"  << std::dec << gettid() << ")" << std::endl;
        if constexpr (std::is_void_v<R>) {
            future.wait();
            // std::cout << "Promise fulfilled [" << &promise << "] (thread 0x" << std::hex << std::this_thread::get_id() 
            //          << "/"  << std::dec << gettid() << ")" << std::endl;
            return;
        } else { 
           R ret = future.get();
        //    std::cout << "Promise fulfilled [" << &promise << "] (thread 0x" << std::hex << std::this_thread::get_id() 
        //              << "/"  << std::dec << gettid() << ")" << std::endl;
           return ret;
        }
    }
};
} // namespace alpaka::cuda::detail