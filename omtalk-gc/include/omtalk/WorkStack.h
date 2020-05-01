#ifndef OMTALK_GC_WORKSTACK_H_
#define OMTALK_GC_WORKSTACK_H_

#include <omtalk/Ref.h>
#include <stack>
#include <vector>

namespace omtalk::gc {

class WorkUnit {
public:
  WorkUnit(Ref<void> ref) : ref(ref) {}

  Ref<void> get() const { return ref; }

private:
  Ref<void> ref;
};

static_assert(std::is_trivially_destructible_v<WorkUnit>);

class WorkStack {
public:
  WorkStack() = default;

  void push(WorkUnit unit) { data.push(unit); }

  WorkUnit pop() {
    auto unit = data.top();
    data.pop();
    return unit;
  }

  bool more() const { return !data.empty(); }

  bool empty() const { return data.empty(); }

private:
  std::stack<WorkUnit, std::vector<WorkUnit>> data;
};

} // namespace omtalk::gc

#endif // OMTALK_GC_WORKSTACK_H_
