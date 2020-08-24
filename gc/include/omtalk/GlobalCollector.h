#ifndef OMTALK_GLOBALCOLLECTOR_H
#define OMTALK_GLOBALCOLLECTOR_H

#include <cstddef>
#include <omtalk/Handle.h>
#include <omtalk/Heap.h>
#include <omtalk/Ref.h>
#include <omtalk/Scheme.h>
#include <stack>

namespace omtalk::gc {

template <typename S>
class MemoryManager;

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

  void push(WorkItem<S> ref) { data.push(ref); }

  WorkItem<S> pop() {
    auto ref = data.top();
    data.pop();
    return ref;
  }

  WorkItem<S> top() { return data.top(); }

  bool more() const { return !data.empty(); }

  bool empyty() const { return data.empty(); }

private:
  std::stack<WorkItem<S>> data;
};

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
  bool concurrentEvacuate() noexcept const { return evacuateRegion != nullptr; }

  /// Get the region we are evacuating to.
  Region *getEvacuateRegion() noexcept const { return evacuateRegion; }
  void setEvacuateRegion(Region *region) noexcept {
    evacuateRegion = region;
    evacuateEnd = region.heapEnd();
  }

  /// Get the address we are evacuating to.
  std::byte *getEvacuateTo() noexcept const { return evacuateAddress; }

  /// Get the address we are evacuating to.
  std::byte *getEvacuateEnd() noexcept const { return evacuateAddress; }

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
// Mark
//===----------------------------------------------------------------------===//

template <typename S>
struct Mark {
  void operator()(GlobalCollectorContext<S> context,
                  ObjectProxy<S> target) noexcept {
    auto ref = target.asRef();
    auto region = Region::get(ref);
    std::cout << "!!! mark: " << ref;
    if (region->mark(ref)) {
      std::cout << " pushed ";
      context.getCollector().getStack().push(target);
    }
    std::cout << std::endl;
  }
};

template <typename S>
void mark(GlobalCollectorContext<S> &context, ObjectProxy<S> target) noexcept {
  Mark<S>()(context, target);
}

template <typename S>
void mark(GlobalCollectorContext<S> &context, Ref<void> target) noexcept {
  Mark<S>()(context, ObjectProxy<S>(target));
}

//===----------------------------------------------------------------------===//
// Scan
//===----------------------------------------------------------------------===//

template <typename S>
class ScanVisitor {
public:
  // Mark any kind of slot proxy
  template <typename SlotProxyT>
  void visit(SlotProxyT slot, GlobalCollectorContext<S> &context) {
    mark<S>(context, proxy::load<RELAXED>(slot));
  }
};

template <typename S>
struct Scan {
  void operator()(GlobalCollectorContext<S> &context,
                  ObjectProxy<S> target) const noexcept {
    ScanVisitor<S> visitor;
    walk<S>(context, target, visitor);
  }
};

template <typename S>
void scan(GlobalCollectorContext<S> &context, ObjectProxy<S> target) noexcept {
  return Scan<S>()(context, target);
}

//===----------------------------------------------------------------------===//
// Sweep
//===----------------------------------------------------------------------===//

template <typename S>
void sweep(GlobalCollectorContext<S> &context, Region &region,
           FreeList &freeList) {
  std::byte *address = region.heapBegin();
  for (const auto object : RegionMarkedObjects<S>(region)) {
    std::byte *objectAddress = reinterpret<std::byte>(object.asRef()).get();
    std::cout << "!!! region: " << region.heapBegin() << std::endl;
    if (address < objectAddress) {
      std::size_t size = objectAddress - address;
      std::cout << "!!! add to freelist: " << address << " size: " << size
                << std::endl;
      freeList.add(address, size);
    }
    address = objectAddress + object.getSize();
  }
  if (address != region.heapEnd()) {
    std::size_t size = region.heapEnd() - address;
    std::cout << "!!! add region tail to free list " << address
              << " size: " << size << std::endl;
  }
}

//===----------------------------------------------------------------------===//
// CopyForward
//===----------------------------------------------------------------------===//

// /// Represents a forwarded object. A forwarded object is one that used to be
// /// at this address but has moved to another address.  This record is left
// /// behind to provide a forwarding address to the object's new location.
// class ForwardedObject {
// public:
//   /// Place a forwarded object at an address.  This will forward to a new
//   /// address.
//   static Ref<ForwardedObject> create(Ref<void> address, Ref<void> to)
//   noexcept {
//     return new (address.get()) ForwardedObject(to);
//   }
//
//   /// Get a ForwardedObject which already exists at an address.
//   static Ref<ForwardedObject> at(Ref<void> address) noexcept {
//     return address.reinterpret<ForwardedObject>();
//   }
//
//   /// Get the forwarding address.  This is the address that the object has
//   moved
//   /// to.
//   Ref<void> getForwardedAddress() const noexcept { return to; }
//
// private:
//   ForwardedObject(Ref<void> to) : to(to) {}
//   Ref<void> to;
// };

/// Returns whether the forward operation was a success. A successful forward
/// operation indicates that all objects were successfully forwarded. A forward
/// operation may copy some objects but still fail if not all objects were
/// copied.
struct CopyForwardResult {
public:
  /// Create a successful forward operation.  The address is the address after
  /// the last copied object.
  static CopyForwardResult success(std::byte *from, std::byte *to) {
    return CopyForwardResult(true, from, to);
  }

  /// Create a failed forwarded operation. The address is the address after the
  /// last copied object. If no object was copied, the address is the
  /// original destination address.
  static CopyForwardResult fail(std::byte *from, std::byte *to) {
    return CopyForwardResult(false, from, to);
  }

  /// Returns if the operation was a success or failure.
  operator bool() { return result; }

  /// Gets the first address after the last forwarded object
  std::byte *getFrom() { return from; }

  /// Gets the first unused address after the forwarded objects.
  std::byte *getTo() { return to; }

private:
  CopyForwardResult(bool result, std::byte *from, std::byte *to)
      : result(result), from(from), to(to) {}

  bool result;
  std::byte *from;
  std::byte *to;
};

/// Default copy forward implemenation.  Assumes that it is safe to memcpy the
/// object and that the object size will not change after copy forward.
template <typename S>
class CopyForward {
public:
  CopyForwardResult operator()(GlobalCollectorContext<S> &context,
                               ObjectProxy<S> from, std::byte *to,
                               std::byte *end) const noexcept {
    std::cout << "!!! forward " << from.asRef().get() << " to " << to
              << std::endl;
    auto forwardedSize = from.getForwardedSize();
    if (forwardedSize > (end - to)) {
      return CopyForwardResult::fail(to);
    }
    memcpy(to, from.asRef().get(), from.getSize());
    from.forward(to);
    return CopyForwardResult::success(from + forwardedSize, to + forwardedSize);
  }
};

template <typename S>
CopyForwardResult copyForward(GlobalCollectorContext<S> &context,
                              ObjectProxy<S> from, std::byte *to,
                              std::byte *end) {
  return CopyForward<S>()(context, from, to, end);
}

template <typename S>
CopyForwardResult copyForward(GlobalCollectorContext<S> &context,
                              Region &fromRegion, std::byte *fromBegin,
                              std::byte *fromEnd, std::byte *toBegin,
                              std::byte toEnd) {
  CopyForwardResult result = CopyForwardResult::fail(fromBegin, toBegin);
  for (const auto object : RegionMarkedObjects<S>(fromBegin, fromEnd)) {
    if (object.isForwarded()) {
      result = copyForward<S>(context, object, toBegin, toEnd);
      toBegin = result.getTo();
      if (!result) {
        break;
      }
    }
  }
  return result;
}

template <typename S>
CopyForwardResult copyForward(GlobalCollectorContext<S> &context, Region &from,
                              Region &to) {
  return copyForward<S>(context, from, from.heapBegin(), from.heapEnd(),
                        to.heapBegin(), to.heapEnd());
}

//===----------------------------------------------------------------------===//
// Fixup
//===----------------------------------------------------------------------===//

template <typename S, typename SlotProxyT>
void fixupSlot(GlobalCollector<S> &context, SlotProxyT slot, Ref<void> to) {
  proxy::store<RELAXED>(slot, to);
}

/// If a slot is pointing to a fowarded object, update the slot to point to
/// the new address of the object
template <typename S, typename SlotProxyT>
void fixupSlot(GlobalCollectorContext<S> &context, SlotProxyT slot) noexcept {
  auto ref = proxy::load<RELAXED>(slot);
  std::cout << "!!! fixup slot " << ref;
  if (inEvacuatedRegion(ref)) {
    auto forwardedAddress = ObjectProxy<S>.getForwa auto forwardedAddress =
        ForwardedObject::at(ref)->getForwardedAddress();
    std::cout << " to " << forwardedAddress;
    fixupSlot(context, slot, forwardedAddress);
  }
  std::cout << std::endl;
}

template <typename S>
class FixupVisitor {
public:
  template <typename SlotProxyT>
  void visit(SlotProxyT slot, GlobalCollectorContext<S> &context) {
    fixupSlot<S>(context, slot);
  }
};

template <typename S>
struct Fixup {
  void operator()(GlobalCollectorContext<S> &context,
                  ObjectProxy<S> target) const noexcept {
    FixupVisitor<S> visitor;
    walk<S>(context, target, visitor);
  }
};

template <typename S>
void fixup(GlobalCollectorContext<S> &context, ObjectProxy<S> target) noexcept {
  return Fixup<S>()(context, target);
}

/// Fix up all objects starting at the beginning of a region.
template <typename S>
void fixup(GlobalCollectorContext<S> &context, Region &region, std::byte *begin,
           std::byte *end) noexcept {
  for (auto object : RegionContiguousObjects<S>(region, begin, end)) {
    fixup<S>(context, object);
  }
}

//===----------------------------------------------------------------------===//
// Evacuate
//===----------------------------------------------------------------------===//

/// Evacuate the object a slot points to and fix up the slot.  This will not
/// detect if the region becomes empty. Does not fix up the evacuated object.
template <typename S, SlotProxy T>
CopyForwardResult evacuate(GlobalCollectorContext<S> &context,
                           SlotProxyT slot) {
  auto to = collector.getEvacuateTo();
  auto end = collector.getEvacuateEnd();
  auto result = copyForward(context, slot.load(), to, end);
  if (!result) {
    // if we ran out of space in the current region, grab a new region to copy
    // in to.
    auto collector = context.getCollector();
    auto regionManager = collector.getRegionManager();
    auto evacuateRegion = collector.getEvacuateRegion();
    regionManager.addRegion(region);
    auto newRegion = regionManager.getEmptyOrNewRegion();
    collector.setEvacuateRegion(newRegion);
    result = copyForward(context, slot.load(), newRegion.heapBegin(),
                         newRegion.heapEnd());
  }
  collector.setEvacuateAddress(result.get());
  fixupSlot(slot, to);
  return result;
}

/// Evacuate an entire region into another region.  Will fixup all region
/// slots.
template <typename S>
CopyForwardResult evacuate(GlobalCollectorContext<S> &context, Region &from,
                           Region &to) {
  auto collector = context.getCollector();
  from.setEvacuated();
  auto result = copyForward<S>(context, from, to);
  fixup<S>(context, to, to.heapBegin(), result.get());
  from.setEvacuated(false);
  getRegionManager.addFreeRegion(region);
  return result;
}

//===----------------------------------------------------------------------===//
// Global Collector Inlines
//===----------------------------------------------------------------------===//

template <typename S>
void GlobalCollector<S>::collect() noexcept {
  GlobalCollectorContext<S> context(this);
  std::cout << "@@@ GC Finalize Previous\n";
  finalCopyForward(context);
  finalFixup(context);
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

template <typename S>
void GlobalCollector<S>::markAllRegionsForEvacuate(Context &Context) noexcept {
  for (auto &region : memoryManager->getRegionManager()) {
    region.setEvacuate();
  }
}

template <typename S>
void GlobalCollector<S>::finalCopyForward(Context &Context) noexcept {
  auto toRegion = evacuateRegion;
  auto toBegin = evacuateBegin;
  auto toEnd = evacuateEnd;
  for (auto &fromRegion : memoryManager->getRegionManager()) {
    if (!fromRegion.isEvacuated())
      continue;
    auto fromBegin = fromRegion.heapBegin();
    auto fromEnd = fromRegion.heapEnd();

    do {
      auto result =
          copyForward(context, fromRegion, fromBegin, fromEnd, toBegin, toEnd);
      if (!result) {
        fromBegin = result.getFrom();
        toRegion = regionManager.getEmptyOrNewRegion();
        toBegin = toRegion.heapBegin();
        toEnd = toRegion.heapEnd();
        regionManager.addRegion(newRegion);
      }
    } while (!result);

    fromBegin = result.getFrom();
    toBegin = result.getTo();
  }

  evacuatedRegion = toRegion;
  evacuatedBegin = toBegin;
  evacuatedEnd = toEnd;
}

template <typename S>
void GlobalCollector<S>::finalFixup(Context &Context) noexcept {
  for (auto &region : memoryManager->getRegionManager()) {
    fixup(region);
  }
}

} // namespace omtalk::gc

#endif