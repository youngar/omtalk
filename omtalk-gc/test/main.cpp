
#include <omtalk/BitArray.h>
#include <omtalk/GC.h>
#include <omtalk/Heap.h>
#include <omtalk/Ref.h>

namespace gc = omtalk::gc;

int main() {
  gc::Collector collector;
  gc::CollectorContext context(collector);
//   Ref<Object> ref = Object::allocate(context);
//   context.collect();
  return 0;
}
