#ifndef OMTALK_DISPATCHER_H
#define OMTALK_DISPATCHER_H

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace omtalk::gc {

class Dispatcher;

enum class WorkerState { INACTIVE, ACTIVE };

class WorkerContext {
public:
  WorkerContext(Dispatcher *dispatcher, unsigned id)
      : dispatcher(dispatcher), id(id) {}

  /// Get the dispatcher associated with this context.
  Dispatcher *getDispatcher() { return dispatcher; }

  /// Return the unique ID of a thread.  Thread IDs range from [0,threadCount].
  unsigned getID() { return id; }

private:
  Dispatcher *dispatcher;
  unsigned id;
};

/// A worker thread.  Has special access to garbage collector internals.
class Worker {
public:
  Worker(Dispatcher *dispatcher, unsigned id)
      : context(dispatcher, id), thread(staticEntry, this) {}

  /// Join the worker thread with the current thread.
  void join() { thread.join(); }

  /// Set the worker to the active state.  This indicates that the worker has a
  /// task to complete.
  void setActive() { state = WorkerState::ACTIVE; }

  bool isActive() { return state == WorkerState::ACTIVE; }

private:
  // Worker thread entry point. Will execute any tasks signaled by the
  // dispatcher.
  static void staticEntry(Worker *worker) noexcept { worker->entry(); }

  void entry() noexcept;

  /// Returns true if the worker should wake up.  A worker should wake up if a
  /// shutdown has been requested, or there is a task.
  bool shouldWakeUp();

  WorkerContext context;
  WorkerState state = WorkerState::INACTIVE;
  std::thread thread;
};

/// A parallel task function.
using Task = std::function<void(WorkerContext &context)>;

/// A thread pool which specializes in running the same task with many threads.
/// Typically will launch all threads on the same function.
class Dispatcher {
public:
  friend Worker;

  /// Create a Dispatcher with as many threads as the hardware supports.
  Dispatcher() : Dispatcher(std::thread::hardware_concurrency()) {}

  /// Create a Dispatcher with workCount number of threads.
  Dispatcher(unsigned workerCount) {
    {
      std::unique_lock lock(workerMutex);
      waitCount = 0;
      shutdown = false;

      workers.reserve(workerCount);
      for (unsigned id = 0; id < workerCount; id++) {
        workers.emplace_back(this, id);
      }
    }
  }

  /// Shutdown the dispatcher, waiting for all workers to exit.Í
  ~Dispatcher() {
    {
      std::unique_lock lock(workerMutex);
      completeCondition.wait(lock, [this] { return isAllWorkComplete(); });
      shutdown = true;
    }
    workerCondition.notify_all();
    for (auto &worker : workers) {
      worker.join();
    }
  }

  // Schedule a task for all threads.  If there is a previous task, this will
  // block until all threads have completed.  This call is not multi-thread
  // safe.
  void run(Task t) {
    std::unique_lock lock(workerMutex);
    completeCondition.wait(lock, [this] { return isAllWorkComplete(); });
    task = t;
    // setting waitCount signals to all threads that there is a new task.
    waitCount = 0;
    for (auto &worker : workers) {
      worker.setActive();
    }
    lock.unlock();
    workerCondition.notify_all();
  }

  /// Get the number of worker threads
  unsigned getWorkerCount() { return workers.size(); }

  /// Returns true if all work is complete
  bool isAllWorkComplete() { return waitCount == getWorkerCount(); }

  /// Wait until all threads have scheduled their current
  void waitForCompletion() {
    std::unique_lock lock(workerMutex);
    completeCondition.wait(lock, [this] { return isAllWorkComplete(); });
  }

private:
  Task getTask() { return task; }

  /// Mutex for worker threads.  Worker condition
  std::mutex workerMutex;

  /// Wakes up workers when there is more work.
  std::condition_variable workerCondition;

  /// Wakes up any thread waiting for work to complete.
  std::condition_variable completeCondition;

  /// The current task for workers to execute.
  Task task;

  /// The number of worker threads waiting.
  unsigned waitCount;

  /// Signals if the worker threads should exit.
  bool shutdown;

  /// List of worker threads.
  std::vector<Worker> workers;
};

inline bool Worker::shouldWakeUp() {
  return isActive() || context.getDispatcher()->shutdown;
}

inline void Worker::entry() noexcept {
  Dispatcher *dispatcher = context.getDispatcher();
  while (true) {
    {
      std::unique_lock lock(dispatcher->workerMutex);
      dispatcher->waitCount++;
      state = WorkerState::INACTIVE;

      // if this is the last thread to finish, notify any thread waiting that
      // work is now complete.
      if (dispatcher->waitCount == dispatcher->getWorkerCount()) {
        dispatcher->completeCondition.notify_all();
      }

      dispatcher->workerCondition.wait(lock, [this] { return shouldWakeUp(); });
    }

    if (dispatcher->shutdown) {
      return;
    }

    // Run the current task with thread local context
    dispatcher->getTask()(context);
  }
}

} // namespace omtalk::gc

#endif
