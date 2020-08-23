#ifndef OMTALK_GLOBALCOLLECTOR_H
#define OMTALK_GLOBALCOLLECTOR_H

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
// Global Collector Scheme -- Default
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

  ~GlobalCollector(){};

  virtual void collect() noexcept override;

  WorkStack<S> &getStack() { return stack; }

  void setup(Context &context) noexcept;

  void scanRoots(Context &context) noexcept;

  void completeScanning(Context &context) noexcept;

  void sweep(Context &context) noexcept;

  void evacuate(Context &context) noexcept;

private:
  MemoryManager<S> *memoryManager;
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
// Mark Functor -- default
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
// Scan Functor -- default
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
void dosweep(GlobalCollectorContext<S> &context, Region &region,
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
// Forward
//===----------------------------------------------------------------------===//

/// Represents a forwarded object.  A forwarded object is one that used to be
/// at this address but has moved to another address.  This record is left
/// behind to provide a forwading address to the object's new location.
class ForwardedObject {
public:
  /// Place a forwarded object at an address.  This will forward to a new
  /// address.
  static Ref<ForwardedObject> create(Ref<void> address, Ref<void> to) noexcept {
    return new (address.get()) ForwardedObject(to);
  }

  /// Get a ForwardedObject which already exists at an address.
  static Ref<ForwardedObject> at(Ref<void> address) noexcept {
    return address.reinterpret<ForwardedObject>();
  }

  /// Get the forwarding address.  This is the address that the object has moved
  /// to.
  Ref<void> getForwardedAddress() const noexcept { return to; }

private:
  ForwardedObject(Ref<void> to) : to(to) {}
  Ref<void> to;
};

/// Returns whether the forward operation was a success.  A successful forward
/// operation indicates that all objects were successfully forwarded.  A forward
/// operation may copy some objects but still fail if not all objects were
/// copied.
struct ForwardResult {
public:
  /// Create a successful forward operation.  The address is the address after
  /// the last copied object.
  static ForwardResult success(std::byte *address) {
    return ForwardResult(true, address);
  }

  /// Create a failed forwarded operation.  The address is the address after the
  /// last copied object.  If no object was copied, the address is the
  /// original destination address.
  static ForwardResult fail(std::byte *address = nullptr) {
    return ForwardResult(false, address);
  }

  /// Returns if the operation was a success or failure.
  operator bool() { return result; }

  /// Gets the first unused address after the forwarded objects.
  std::byte *get() { return address; }

private:
  ForwardResult(bool result, std::byte *address)
      : result(result), address(address) {}

  bool result;
  std::byte *address;
};

template <typename S>
class Forward {
public:
  ForwardResult operator()(GlobalCollectorContext<S> &context,
                           ObjectProxy<S> from, std::byte *to,
                           std::byte *end) const noexcept {
    std::cout << "!!! forward " << from.asRef().get() << " to " << to
              << std::endl;
    auto forwardedSize = from.getForwardedSize();
    if (forwardedSize > (end - to)) {
      return ForwardResult::fail(to);
    }
    memcpy(to, from.asRef().get(), from.getSize());
    ForwardedObject::create(from.asRef(), to);
    return ForwardResult::success(to + forwardedSize);
  }
};

template <typename S>
ForwardResult forward(GlobalCollectorContext<S> &context, ObjectProxy<S> from,
                      std::byte *to, std::byte *end) {
  return Forward<S>()(context, from, to, end);
}

template <typename S>
ForwardResult forward(GlobalCollectorContext<S> &context, Region &from,
                      std::byte *to, std::byte *end) {
  ForwardResult result = ForwardResult::fail(to);
  for (const auto object : RegionMarkedObjects<S>(from)) {
    result = forward<S>(context, object, result.get(), end);
    if (!result) {
      assert(false && "TODO: handle case when not all objects can be evacuated "
                      "to a region");
      break;
    }
  }
  return result;
}

template <typename S>
ForwardResult forward(GlobalCollectorContext<S> &context, Region &from,
                      Region &to) {
  return forward<S>(context, from, to.heapBegin(), to.heapEnd());
}

//===----------------------------------------------------------------------===//
// Fixup
//===----------------------------------------------------------------------===//

/// Returns whether a ref is in a region that has been evacuated during a
/// garbage collection
template <typename T>
bool inEvacuatedRegion(Ref<T> address) {
  return Region::get(address)->isEvacuated();
}

/// If a slot is pointing to a fowarded object, update the slot to point to the
/// new address of the object
template <typename S, typename SlotProxyT>
struct FixupSlot {
  void operator()(GlobalCollectorContext<S> context, SlotProxyT slot) noexcept {
    auto ref = proxy::load<RELAXED>(slot);
    std::cout << "!!! fixup slot " << ref;
    if (inEvacuatedRegion(ref)) {
      auto forwardedAddress = ForwardedObject::at(ref)->getForwardedAddress();
      std::cout << " to " << forwardedAddress;
      proxy::store<RELAXED>(slot, forwardedAddress);
    }
    std::cout << std::endl;
  }
};

template <typename S, typename SlotProxyT>
void fixupSlot(GlobalCollectorContext<S> &context, SlotProxyT slot) noexcept {
  FixupSlot<S, SlotProxyT>()(context, slot);
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

template <typename S>
ForwardResult evacuate(GlobalCollectorContext<S> &context, Region &from,
                       Region &to) {
  from.setEvacuated();
  ForwardResult result = forward<S>(context, from, to);
  fixup<S>(context, to, to.heapBegin(), result.get());
  from.setEvacuated(false);
  return result;
}

//===----------------------------------------------------------------------===//
// Global Collector Inlines
//===----------------------------------------------------------------------===//

template <typename S>
void GlobalCollector<S>::collect() noexcept {
  GlobalCollectorContext<S> context(this);
  std::cout << "@@@ GC Start\n";
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
    dosweep<S>(context, region, freeList);
  }
  memoryManager->setFreeList(freeList);
}

template <typename S>
void GlobalCollector<S>::evacuate(Context &Context) noexcept {
  assert(false && "Not implemented");
}

} // namespace omtalk::gc

#endif