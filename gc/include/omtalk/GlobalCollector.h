#ifndef OMTALK_GLOBALCOLLECTOR_H
#define OMTALK_GLOBALCOLLECTOR_H

#include <cstddef>
#include <omtalk/Copy.h>
#include <omtalk/Handle.h>
#include <omtalk/Heap.h>
#include <omtalk/Mark.h>
#include <omtalk/MarkCompactWorker.h>
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
  void preMark(Context &context) noexcept;

  /// Scan all roots.
  // requires GC safepoint
  void markRoots(Context &context) noexcept;

  /// Concurrent collection helpers
  // @{
  void mark(Context &context) noexcept;

  void postMark(Context &context) noexcept;

  void preCompact(Context &context) noexcept;
  void compact(Context &context) noexcept;
  void postCompact(Context &context) noexcept;

  // Requires GC safepoint
  void fixupRoots(Context &context) noexcept;
  void fixup(Context &context) noexcept;
  void postFixup(Context &context) noexcept;

  void sweep(Context &context) noexcept;
  void evacuate(Context &context) noexcept;
  // @}

  /// Get the global workstack.
  WorkStack<S> &getStack() { return stack; }

private:
  MemoryManager<S> *memoryManager;
  MarkCompactWorker<S> worker;
  WorkStack<S> stack;
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
  preMark(context);
  markRoots(context);
  std::cout << "@@@ starting concurrent\n";
  worker.run();
}

template <typename S>
void GlobalCollector<S>::wait(Context &context) noexcept {
  worker.wait();
}

template <typename S>
void GlobalCollector<S>::preMark(Context &context) noexcept {
  OMTALK_ASSERT(stack.empty());
  auto &regionManager = memoryManager->getRegionManager();

  for (auto &region : regionManager.getRegions()) {
    std::cout << "clearing region " << &region << std::endl;
    region.clearMarkMap();
    region.clearStatistics();
  }

  for (auto &region : regionManager.getAllocateRegions()) {
    std::cout << "clearing region " << &region << std::endl;
    region.clearMarkMap();
    region.clearStatistics();
  }

  memoryManager->enableWriteBarrier();
}

template <typename S>
void GlobalCollector<S>::markRoots(Context &context) noexcept {
  ScanVisitor<S> visitor;
  memoryManager->getRootWalker().walk(context, visitor);
}

template <typename S>
void GlobalCollector<S>::mark(Context &context) noexcept {
  while (stack.more()) {
    auto item = stack.pop();
    std::cout << "!!! Scan: " << item.target.asRef() << std::endl;
    scan<S>(context, item.target);
  }
}

template <typename S>
void GlobalCollector<S>::postMark(Context &context) noexcept {
  memoryManager->disableWriteBarrier();
}

template <typename S>
void GlobalCollector<S>::sweep(Context &context) noexcept {
  FreeList freeList;
  for (auto &region : memoryManager->getRegionManager()) {
    omtalk::gc::sweep<S>(context, region, freeList);
  }
  memoryManager->setFreeList(freeList);
}

template <typename S>
void GlobalCollector<S>::preCompact(Context &context) noexcept {
  // select regions for collection.  Selection is based on the regions with the
  // least amount of live data in them.
  auto &regionManager = memoryManager->getRegionManager();
  auto regionList = regionManager.getRegions();
  auto emptyList = regionManager.getEmptyRegions();
  auto evacList = regionManager.getEvacuateRegions();
  auto iter = regionList.begin();
  auto end = regionList.end();
  while (iter != end) {
    auto &region = *iter;
    // increment the iterator before potentially removing the current region
    // from the list.
    iter++;
    // regions less than half full are candidates for evacuation.
    auto liveData = region.getLiveDataSize();
    if (liveData == 0) {
      std::cout << "!! precompact empty region: " << &region << " evacuate\n";
      std::scoped_lock lock(regionManager);
      regionList.remove(&region);
      emptyList.push_front(&region);
    } else if (liveData < (REGION_SIZE / 2)) {
      std::cout << "!! precompact region: " << &region << " evacuate\n";
      {
        std::scoped_lock lock(regionManager);
        regionList.remove(&region);
        evacList.push_front(&region);
      }
      // populate the forwarding map
      auto liveObjects = RegionMarkedObjectsIterator<S>(region);
      region.getForwardingMap().rebuild(liveObjects,
                                        region.getLiveObjectCount());
      // As soon as this bit is set, mutator threads will start evacuating
      // objects.
      region.setEvacuating(true);
    }
  }
}

template <typename S>
void GlobalCollector<S>::compact(Context &context) noexcept {
  auto &regionManager = memoryManager->getRegionManager();
  // auto regionList = regionManager.getRegions();
  // auto emptyList = regionManager.getEmptyRegions();
  auto evacList = regionManager.getEvacuateRegions();

  Region *toRegion;
  {
    std::scoped_lock lock(regionManager);
    toRegion = regionManager.getEmptyOrNewRegion();
  }
  auto *to = toRegion->heapBegin();
  auto size = toRegion->size();

  for (auto &fromRegion : evacList) {
    std::cout << "!!! compact fromRegion " << &fromRegion << std::endl;
    ForwardingMap &map = fromRegion.getForwardingMap();

    for (auto object : RegionMarkedObjects<S>(fromRegion)) {
      std::cout << "!!! compact object " << object.asRef() << std::endl;
      auto &entry = map[object];
      if (!entry.tryLock()) {
        // object has already been copied
        continue;
      }
      auto result = copy(context, object, to, size);
      if (!result) {
        // get a new region and copy again
        {
          std::scoped_lock lock(regionManager);
          toRegion = regionManager.getEmptyOrNewRegion();
          OMTALK_ASSERT(toRegion);
        }
        to = toRegion->heapBegin();
        size = toRegion->size();
        result = copy(context, object, to, size);
        OMTALK_ASSERT(result == true);
      }
      entry.set(to);
      to += result.getCopySize();
      size -= result.getCopySize();
    }
  }
}

template <typename S>
void GlobalCollector<S>::postCompact(Context &context) noexcept {
  RegionManager &regionManager = memoryManager->getRegionManager();
  for (auto &region : regionManager.getEvacuateRegions()) {
    region.getForwardingMap().clear();
    region.setEvacuating(false);
  }
}

template <typename S>
void GlobalCollector<S>::fixupRoots(Context &context) noexcept {
  // FixupVisitor<S> visitor;
  // memoryManager->getRootWalker().walk(context, visitor);
}

template <typename S>
void GlobalCollector<S>::fixup(Context &context) noexcept {

  // Regular region fixup
  // RegionManager &regionManager = memoryManager->getRegionManager();
  // auto &regionList = regionManager.getRegions();
  // for (auto &region : regionList) {
  //   for (auto object : RegionMarkedObjects(*region)) {
  //     fixup(context, object);
  //   }
  // }

  // Evacuate regions do not have an up to date markmap, and must be walked
  // using the contiguous heap iterator.
  // auto &evacList = regionManager.getEvacuateRegions();
  // for (auto &region : evacList) {
  //   for (auto object : RegionContiguousObjects(*region)) {
  //     fixup(context, object);
  //   }
  // }
}

template <typename S>
void GlobalCollector<S>::postFixup(Context &context) noexcept {
  // Move evacuate regions into the regular region list
  RegionManager &regionManager = memoryManager->getRegionManager();
  std::scoped_lock lock(regionManager);
  regionManager.getEmptyRegions().splice(regionManager.getEvacuateRegions());
}

} // namespace omtalk::gc

#endif