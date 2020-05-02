
#include <iostream>
#include <omtalk/BitArray.h>
#include <omtalk/GC.h>
#include <omtalk/Heap.h>
#include <omtalk/Ref.h>

namespace gc = omtalk::gc;

struct Object {
  std::size_t size;
};

std::ostream &operator<<(std::ostream &out, const Object &obj) {
  out << "(Object";
  out << " size: " << obj.size;
  out << ")";
  return out;
}

int main() {
  gc::Collector collector;
  gc::CollectorContext context(collector);

  do {
    gc::Ref<Object> ref = context.allocate<Object>(gc::MIN_OBJECT_SIZE);
    if (ref == nullptr) {
      break;
    }
    ref->size = gc::MIN_OBJECT_SIZE;
    std::cout << "successful allocation\n" << ref << std::endl;
  } while (true);

  return 0;
}
