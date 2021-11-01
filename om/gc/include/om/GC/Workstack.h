#ifndef OM_GC_WORKSTACK_H
#define OM_GC_WORKSTACK_H

#include <mutex>
#include <om/GC/Ref.h>
#include <om/GC/Scheme.h>
#include <stack>

namespace om::gc {

//===----------------------------------------------------------------------===//
// Work Stack
//===----------------------------------------------------------------------===//

template <typename S>
struct WorkItem {
public:
  WorkItem(ObjectProxy<S> target) : target(target) {}

  ObjectProxy<S> target;
};

template <typename S>
class WorkStack {
public:
  WorkStack() = default;

  void push(WorkItem<S> ref) {
    std::unique_lock lock(stackMutex);
    data.push_back(ref);
  }

  WorkItem<S> pop() {
    std::unique_lock lock(stackMutex);
    auto ref = data.back();
    data.pop_back();
    return ref;
  }

  WorkItem<S> top() {
    std::unique_lock lock(stackMutex);
    return data.back();
  }

  bool more() {
    std::unique_lock lock(stackMutex);
    return !data.empty();
  }

  bool empty() {
    std::unique_lock lock(stackMutex);
    return data.empty();
  }

  void clear() {
    std::unique_lock lock(stackMutex);
    data.clear();
  }

private:
  std::mutex stackMutex;
  std::vector<WorkItem<S>> data;
};

} // namespace om::gc

#endif