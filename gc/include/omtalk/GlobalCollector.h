#ifndef OMTALK_GLOBALCOLLECTOR_H
#define OMTALK_GLOBALCOLLECTOR_H

#include <cstddef>
#include <omtalk/Handle.h>
#include <omtalk/Heap.h>
#include <omtalk/Mark.h>
#include <omtalk/Ref.h>
#include <omtalk/Scheme.h>
#include <omtalk/Sweep.h>
#include <stack>

namespace omtalk::gc {

template <typename S>
class MemoryManager;

//===----------------------------------------------------------------------===//
// Global Collector
//===----------------------------------------------------------------------===//

/// Interface for the global collector scheme.
class AbstractGlobalCollector {
public:
  virtual void collect() = 0;

  virtual ~AbstractGlobalCollector() = default;
};

template <typename S>
class GlobalCollectorContext;

/// Default implementation of global collection.
template <typename S>
class GlobalCollector : public AbstractGlobalCollector {
public:
  using Context = GlobalCollectorContext<S>;

  GlobalCollector(MemoryManager<S> *memoryManager)
      : memoryManager(memoryManager) {}

  ~GlobalCollector() {}

  virtual void collect() noexcept override;

  WorkStack<S> &getStack() { return stack; }

  void setup(Context &context) noexcept;
  void scanRoots(Context &context) noexcept;
  void completeScanning(Context &context) noexcept;
  void sweep(Context &context) noexcept;
  void evacuate(Context &context) noexcept;

  /// Finish a concurrent evacuation.
  void finalEvacuate(Context &context) noexcept;

  /// Returns true if concurrent evacuation is on-going
  bool concurrentEvacuate() const noexcept { return evacuateRegion != nullptr; }

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
void GlobalCollector<S>::collect() noexcept {
  GlobalCollectorContext<S> context(this);
  // std::cout << "@@@ GC Finalize Previous\n";
  // finalCopyForward(context);
  // finalFixup(context);
  std::cout << "@@@ GC Roots\n";
  setup(context);
  scanRoots(context);
  std::cout << "@@@ GC Marking\n";
  completeScanning(context);
  std::cout << "@@@ GC Sweep\n";
  sweep(context);
  std::cout << "@@@ GC End\n";
}

template <typename S>
void GlobalCollector<S>::setup(Context &context) noexcept {
  assert(stack->empty());
  memoryManager->getRegionManager().clearMarkMaps();
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

// template <typename S>
// void GlobalCollector<S>::markAllRegionsForEvacuate(Context &Context) noexcept
// {
//   for (auto &region : memoryManager->getRegionManager()) {
//     region.setEvacuate();
//   }
// }

// template <typename S>
// void GlobalCollector<S>::finalCopyForward(Context &Context) noexcept {
//   auto toRegion = evacuateRegion;
//   auto toBegin = evacuateBegin;
//   auto toEnd = evacuateEnd;
//   for (auto &fromRegion : memoryManager->getRegionManager()) {
//     if (!fromRegion.isEvacuated())
//       continue;
//     auto fromBegin = fromRegion.heapBegin();
//     auto fromEnd = fromRegion.heapEnd();

//     do {
//       auto result =
//           copyForward(context, fromRegion, fromBegin, fromEnd, toBegin,
//           toEnd);
//       if (!result) {
//         fromBegin = result.getFrom();
//         toRegion = regionManager.getEmptyOrNewRegion();
//         toBegin = toRegion.heapBegin();
//         toEnd = toRegion.heapEnd();
//         regionManager.addRegion(newRegion);
//       }
//     } while (!result);

//     fromBegin = result.getFrom();
//     toBegin = result.getTo();
//   }

//   evacuatedRegion = toRegion;
//   evacuatedBegin = toBegin;
//   evacuatedEnd = toEnd;
// }

// template <typename S>
// void GlobalCollector<S>::finalFixup(Context &Context) noexcept {
//   for (auto &region : memoryManager->getRegionManager()) {
//     fixup(region);
//   }
// }

} // namespace omtalk::gc

#endif