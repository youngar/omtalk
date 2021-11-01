#ifndef OM_GC_GLOBALCOLLECTOR_H
#define OM_GC_GLOBALCOLLECTOR_H

#include <ab/Util/Assert.h>
#include <cstddef>
#include <om/GC/Copy.h>
#include <om/GC/Fixup.h>
#include <om/GC/Handle.h>
#include <om/GC/Heap.h>
#include <om/GC/Mark.h>
#include <om/GC/MarkCompactWorker.h>
#include <om/GC/MarkFixup.h>
#include <om/GC/Ref.h>
#include <om/GC/Scheme.h>
#include <om/GC/Sweep.h>
#include <om/GC/Workstack.h>
#include <stack>

namespace om::gc {

template <typename S>
class GlobalCollectorContext;

template <typename S>
class MemoryManager;

template <typename S>
class ScanVisitor;

template <typename S>
class MarkFixupVisitor;

template <typename S>
class MarkCompactWorker;

template <typename S>
void scan(GlobalCollectorContext<S> &context, ObjectProxy<S> target) noexcept;

template <typename S>
void sweep(GlobalCollectorContext<S> &context, Region &region,
           FreeList &freeList);

template <typename S>
void markFixup(GlobalCollectorContext<S> &context,
               ObjectProxy<S> target) noexcept;

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

  /// Prepare a region for evacuation and move it to the evacuating list.  After
  /// this the load barrier may evacuate objects from the region.
  void selectForEvacuate(Context &context, Region &region) noexcept;

  void preCompact(Context &context) noexcept;
  void compact(Context &context) noexcept;
  void postCompact(Context &context) noexcept;

  void sweep(Context &context) noexcept;
  // @}

  /// Get the MemoryManager
  MemoryManager<S> *getMemoryManager() noexcept { return memoryManager; }

  /// Get the global workstack.
  WorkStack<S> &getStack() noexcept { return stack; }

private:
  MemoryManager<S> *memoryManager;
  MarkCompactWorker<S> worker;
  WorkStack<S> stack;
};

/// Default context of the marking scheme.
template <typename S>
class GlobalCollectorContext {
public:
  explicit GlobalCollectorContext(GlobalCollector<S> *collector) noexcept
      : collector(collector) {}

  MemoryManager<S> &getMemoryManager() noexcept {
    return *collector->getMemoryManager();
  }

  GlobalCollector<S> &getCollector() const { return *collector; }

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
  AB_ASSERT(stack.empty());

  auto &regionManager = memoryManager->getRegionManager();

  for (auto &region : regionManager.getRegions()) {
    std::cout << "!!! clear markmap for regular region " << &region
              << std::endl;
    region.clearMarkMap();
    region.clearStatistics();
  }

  for (auto &region : regionManager.getAllocateRegions()) {
    std::cout << "!!! clear markmap for allocate region " << &region
              << std::endl;
    region.clearMarkMap();
    region.clearStatistics();
  }

  for (auto &region : regionManager.getEvacuateRegions()) {
    std::cout << "!!! clear markmap for evactuate region " << &region
              << std::endl;
    region.clearMarkMap();
    region.clearStatistics();
  }

  memoryManager->enableMarking();
}

template <typename S>
void GlobalCollector<S>::markRoots(Context &context) noexcept {
  MarkFixupVisitor<S> visitor;
  memoryManager->getRootWalker().walk(context, visitor);
}

template <typename S>
void GlobalCollector<S>::mark(Context &context) noexcept {
  while (stack.more()) {
    auto item = stack.pop();
    std::cout << "!!! Scan: " << item.target.asRef() << std::endl;
    markFixup<S>(context, item.target);
  }
}

template <typename S>
void GlobalCollector<S>::postMark(Context &context) noexcept {
  memoryManager->disableMarking();
}

template <typename S>
void GlobalCollector<S>::sweep(Context &context) noexcept {
  FreeList freeList;
  for (auto &region : memoryManager->getRegionManager()) {
    om::gc::sweep<S>(context, region, freeList);
  }
  memoryManager->setFreeList(freeList);
}

template <typename S>
void GlobalCollector<S>::selectForEvacuate(Context &context,
                                           Region &region) noexcept {
  auto &regionManager = memoryManager->getRegionManager();
  auto &regionList = regionManager.getRegions();
  auto &evacList = regionManager.getEvacuateRegions();

  std::cout << "!! select for compact region: " << &region << "\n";
  {
    std::scoped_lock lock(regionManager);
    regionList.remove(&region);
    evacList.push_front(&region);
  }

  // populate the forwarding map
  auto liveObjects = RegionMarkedObjectsIterator<S>(region);
  region.getForwardingMap().rebuild(liveObjects, region.getLiveObjectCount());

  // As soon as this bit is set, mutator threads will start evacuating
  // objects.
  region.setEvacuating(true);
}

template <typename S>
void GlobalCollector<S>::preCompact(Context &context) noexcept {
  // select regions for collection.  Selection is based on the regions with the
  // least amount of live data in them.
  auto &regionManager = memoryManager->getRegionManager();
  auto &regionList = regionManager.getRegions();
  auto &emptyList = regionManager.getEmptyRegions();
  auto &evacList = regionManager.getEvacuateRegions();

  // We may now start reusing evacuate regions to allocate out of
  {
    std::scoped_lock lock(regionManager);
    regionList.splice(evacList);
  }

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
      selectForEvacuate(context, region);
    } else {
      std::cout << "!!! precompact clear forwarding map " << &region
                << " evacuate\n";
      region.getForwardingMap().clear();
      region.setEvacuating(false);
    }
  }
}

template <typename S>
void GlobalCollector<S>::compact(Context &context) noexcept {
  auto &regionManager = memoryManager->getRegionManager();
  auto &evacList = regionManager.getEvacuateRegions();

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
        std::cout << "!!! previously copied to: " << entry.get() << std::endl;
        continue;
      }
      auto result = copy(context, object, to, size);
      if (!result) {
        // get a new region and copy again
        {
          std::scoped_lock lock(regionManager);
          toRegion = regionManager.getEmptyOrNewRegion();
          AB_ASSERT(toRegion);
        }
        to = toRegion->heapBegin();
        size = toRegion->size();
        result = copy(context, object, to, size);
        AB_ASSERT(result == true);
      }
      entry.set(to);
      std::cout << "!!! copied to: " << entry.get() << std::endl;
      to += result.getCopySize();
      size -= result.getCopySize();
    }
  }
}

template <typename S>
void GlobalCollector<S>::postCompact(Context &context) noexcept {}

} // namespace om::gc

#endif