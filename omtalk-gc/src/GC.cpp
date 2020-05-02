#include <iostream>
#include <omtalk/GC.h>

using namespace omtalk;
using namespace omtalk::gc;

//===----------------------------------------------------------------------===//
// Collector
//===----------------------------------------------------------------------===//

bool Collector::refreshBuffer(CollectorContext *cx, std::size_t minimumSize) {

  std::cout << "getting thing1\n";
  // search the free list for an entry at least as big
  FreeBlock *block = freeList.firstFit(minimumSize);
  if (block != nullptr) {
    cx->buffer().begin = block->begin();
    cx->buffer().end = block->end();
    return true;
  }

  std::cout << "getting thing2\n";
  // Get a new region
  Region *region = regionManager.allocateRegion();
  if (region != nullptr) {
    std::cout << "begin " << region->heapBegin() << "end" << region->heapEnd()
              << std::endl;
    cx->buffer().begin = region->heapBegin();
    cx->buffer().end = region->heapEnd();

    return true;
  }

  std::cout << "failed\n";
  // Failed to allocate
  return false;
};

//===----------------------------------------------------------------------===//
// CollectorContext
//===----------------------------------------------------------------------===//

bool CollectorContext::refreshBuffer(std::size_t minimumSize) {
  return collector->refreshBuffer(this, minimumSize);
}
