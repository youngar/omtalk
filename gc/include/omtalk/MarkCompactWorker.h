#ifndef OMTALK_COMPACTWORKER_H
#define OMTALK_COMPACTWORKER_H

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <omtalk/GlobalCollector.h>
#include <thread>

namespace omtalk::gc {

template <typename S>
class GlobalCollectorContext;

template <typename S>
class GlobalCollector;

template <typename S>
class MarkCompactWorker {
public:
  MarkCompactWorker(GlobalCollector<S> *gc)
      : context(gc), gc(gc), thread(staticEntry, this) {}

  ~MarkCompactWorker() {
    // Signal to the thread to shutdown and wake it up.
    std::unique_lock lock(stateMutex);
    state = State::SHUTDOWN;
    lock.unlock();
    workCondition.notify_all();
    thread.join();
  }

  /// Wait for the completion of a GC cycle
  void wait() {
    std::unique_lock lock(stateMutex);
    completeCondition.wait(lock, [&] { return state != State::ACTIVE; });
  }

  /// Execute a garbage collection.  Returns true if a new collection was
  /// started. Returns false if a collection is already in progress.
  bool run() {
    std::unique_lock lock(stateMutex);
    if (state == State::INACTIVE) {
      state = State::ACTIVE;
      lock.unlock();
      workCondition.notify_all();
      return true;
    }
    return false;
  }

private:
  enum class State { INACTIVE, ACTIVE, SHUTDOWN };

  static void staticEntry(MarkCompactWorker *t) { t->entry(); }

  void entry() {
    std::unique_lock lock(stateMutex);
    while (true) {
      // wait for more work to appear
      workCondition.wait(lock, [&] { return state != State::INACTIVE; });

      // if shutdown is requested, have the thread exit.
      if (state == State::SHUTDOWN) {
        return;
      }

      if (state == State::ACTIVE) {
        lock.unlock();
        collect();
        lock.lock();
        state = State::INACTIVE;
        lock.unlock();
        completeCondition.notify_all();
        lock.lock();
      }
    }
  }

  void collect() {
    std::cout << "### background start\n";
    std::cout << "### background mark\n";
    gc->mark(context);
    std::cout << "### background post-mark\n";
    gc->postMark(context);
    std::cout << "### background pre-compact\n";
    gc->preCompact(context);
    std::cout << "### background compact\n";
    gc->compact(context);
    std::cout << "### background post-compact\n";
    gc->postCompact(context);
    std::cout << "### backgound GC complete\n";
  }

  GlobalCollectorContext<S> context;
  GlobalCollector<S> *gc;

  /// The current state of the thread.
  State state = State::INACTIVE;

  /// Mutex guarding all requests
  std::mutex stateMutex;

  // Condition triggered when there is work available
  std::condition_variable workCondition;

  // Condition triggered when work is complete
  std::condition_variable completeCondition;

  std::thread thread;
};

} // namespace omtalk::gc

#endif