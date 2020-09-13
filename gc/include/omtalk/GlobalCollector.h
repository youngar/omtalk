#ifndef OMTALK_GLOBALCOLLECTOR_H
#define OMTALK_GLOBALCOLLECTOR_H

#include <cstddef>
#include <omtalk/MarkCompactWorker.h>
#include <omtalk/Handle.h>
#include <omtalk/Heap.h>
#include <omtalk/Mark.h>
#include <omtalk/Ref.h>
#include <omtalk/Scheme.h>
#include <omtalk/Sweep.h>
#include <omtalk/Util/Assert.h>
#include <omtalk/Workstack.h>
#include <stack>

namespace omtalk::gc {

template <typename S>
class GlobalCollectorContext;

template <typename S>
class MemoryManager;

template <typename S>
class ScanVisitor;

template <typename S>
class MarkCompactWorker;

template <typename S>
void scan(GlobalCollectorContext<S> &context, ObjectProxy<S> target) noexcept;

template <typename S>
void sweep(GlobalCollectorContext<S> &context, Region &region,
           FreeList &freeList);

//===----------------------------------------------------------------------===//
// Global Collector
//===----------------------------------------------------------------------===//

template <typename S>
class GlobalCollectorContext;

/// Default implementation of global collection.
template <typename S>
class GlobalCollector {
public:
  using Context = GlobalCollectorContext<S>;

  GlobalCollector(MemoryManager<S> *memoryManager)
      : memoryManager(memoryManager), worker(this) {}

  ~GlobalCollector() {}

  /// Explicit garbage collection.  Will cancel the previous garbage collection
  /// and execute a full GC cycle.
  void collect(Context &contex) noexcept;

  /// Kickoff a new GC if one is currently not running.
  void kickoff(Context &context) noexcept;

  /// Wait for the current GC to finish.
  void wait(Context &context) noexcept;

  /// Perform a garbage collection.  Will return as early as possible.
  void concurrentCollect(Context &context) noexcept;

  /// Cancel an in-progress GC.  This is used to restart a garbage collection
  /// for a new global gc.
  void cancel(Context &context) noexcept;

  /// Set up for the next garbage collection. Must only be called after the
  /// previous GC has finished.
  void setup(Context &context) noexcept;

  /// Scan all roots.
  void scanRoots(Context &context) noexcept;

  /// Concurrent collection helpers
  // @{
  void completeScanning(Context &context) noexcept;
  void sweep(Context &context) noexcept;
  void evacuate(Context &context) noexcept;
  // @}

  /// Finish a concurrent evacuation.
  void finalEvacuate(Context &context) noexcept;

  /// Returns true if concurrent evacuation is on-going
  bool concurrentEvacuate() const noexcept { return evacuateRegion != nullptr; }

  /// Get the global workstack.
  WorkStack<S> &getStack() { return stack; }

  /// Get the region we are evacuating to.
  Region *getEvacuateRegion() const noexcept { return evacuateRegion; }

  void setEvacuateRegion(Region *region) noexcept {
    evacuateRegion = region;
    evacuateEnd = region->heapEnd();
  }

  /// Get the address we are evacuating to.
  std::byte *getEvacuateTo() const noexcept { return evacuateTo; }

  /// Get the address we are evacuating to.
  std::byte *getEvacuateEnd() const noexcept { return evacuateEnd; }

  /// Get the region we are evacuating to.
  void setEvacuateTo(std::byte *to) noexcept { evacuateTo = to; }

private:
  MemoryManager<S> *memoryManager;
  MarkCompactWorker<S> worker;
  WorkStack<S> stack;

  /// The region to evacuate objects to.
  ///
  /// TODO: This system is a placeholder for something more robust and thread
  /// safe
  Region *evacuateRegion = nullptr;
  std::byte *evacuateTo = nullptr;
  std::byte *evacuateEnd = nullptr;
};

/// Default context of the marking scheme.
template <typename S>
class GlobalCollectorContext {
public:
  explicit GlobalCollectorContext(GlobalCollector<S> *collector)
      : collector(collector) {}

  GlobalCollector<S> &getCollector() { return *collector; }

private:
  GlobalCollector<S> *collector;
};

//===----------------------------------------------------------------------===//
// Global Collector Inlines
//===----------------------------------------------------------------------===//

template <typename S>
void GlobalCollector<S>::collect(Context &context) noexcept {
  kickoff(context);
  worker.wait();
  std::cout << "@@@ GC End\n";
}

template <typename S>
void GlobalCollector<S>::kickoff(Context &context) noexcept {
  std::cout << "@@@ GC Roots\n";
  setup(context);
  scanRoots(context);
  std::cout << "@@@ starting concurrent\n";
  worker.run();
}

template <typename S>
void GlobalCollector<S>::wait(Context &context) noexcept {
  worker.wait();
}

template <typename S>
void GlobalCollector<S>::setup(Context &context) noexcept {
  // Clear out any marked references.  This can especially happen from
  // allocations.  Theoretically if the previous cycle completes, the stacks
  // should be empty.
  stack.clear();

  auto &regionManager = memoryManager->getRegionManager();
  regionManager.clearMarkMaps();

  // select regions for collection.  Selection is based on the regions with the
  // least amount of live data in them.
  for (auto &region : regionManager) {

    // regions less than half full are candidates for evacuation.
    auto liveData = region.getLiveDataSize();
    if (liveData < (REGION_SIZE / 2)) {
      region.setEvacuating(true);
    } else {
      region.setEvacuating(false);
    }

    // reset region statistics
    region.clearLiveDataSize();
  }
}

template <typename S>
void GlobalCollector<S>::scanRoots(Context &context) noexcept {
  ScanVisitor<S> visitor;
  memoryManager->getRootWalker().walk(context, visitor);
}

template <typename S>
void GlobalCollector<S>::completeScanning(Context &context) noexcept {
  while (stack.more()) {
    auto item = stack.pop();
    std::cout << "!!! Scan: " << item.target.asRef() << std::endl;
    scan<S>(context, item.target);
  }
}

template <typename S>
void GlobalCollector<S>::sweep(Context &context) noexcept {
  FreeList freeList;
  for (auto &region : memoryManager->getRegionManager()) {
    omtalk::gc::sweep<S>(context, region, freeList);
  }
  memoryManager->setFreeList(freeList);
}
} // namespace omtalk::gc

#endif