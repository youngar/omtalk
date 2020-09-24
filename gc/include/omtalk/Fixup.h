#ifndef OMTALK_FIXUP_H
#define OMTALK_FIXUP_H

#include <omtalk/Heap.h>
#include <omtalk/Record.h>
#include <omtalk/Ref.h>

namespace omtalk::gc {

template <typename S>
class GlobalCollector;

template <typename S, typename SlotProxyT>
void fixupSlot(GlobalCollectorContext<S> &context, SlotProxyT slot,
               Ref<void> to) {
  proxy::store<RELAXED>(slot, to);
}

/// If a slot is pointing to a fowarded object, update the slot to point to
/// the new address of the object.
template <typename S, typename SlotProxyT>
void fixupSlot(GlobalCollectorContext<S> &context, SlotProxyT slot) noexcept {
  auto ref = proxy::load<RELAXED>(slot);
  auto address = ref.get();
  auto *region = Region::get(ref);
  if (region->isEvacuating()) {
    auto &entry = region->getForwardingMap()[address];
    auto forwardedAddress = entry.get();
    std::cout << "!!! fixup slot " << ref << " to " << forwardedAddress
              << std::endl;
    fixupSlot(context, slot, makeRef(forwardedAddress));
  }
}

/// Call fixupSlot on every slot visited.
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
    record<S>(context, target);
    walk<S>(context, target, visitor);
  }
};

/// Fix up all slots in an object
template <typename S>
void fixup(GlobalCollectorContext<S> &context, ObjectProxy<S> target) noexcept {
  return Fixup<S>()(context, target);
}

} // namespace omtalk::gc

#endif