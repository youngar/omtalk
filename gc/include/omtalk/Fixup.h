#ifndef OMTALK_FIXUP_H
#define OMTALK_FIXUP_H

namespace omtalk::gc {

template <typename S, typename SlotProxyT>
void fixupSlot(GlobalCollector<S> &context, SlotProxyT slot, Ref<void> to) {
  proxy::store<RELAXED>(slot, to);
}

/// If a slot is pointing to a fowarded object, update the slot to point to
/// the new address of the object.
template <typename S, typename SlotProxyT>
void fixupSlot(GlobalCollectorContext<S> &context, SlotProxyT slot) noexcept {
  auto ref = proxy::load<RELAXED>(slot);
  std::cout << "!!! fixup slot " << ref;
  if (inEvacuatedRegion(ref)) {
    auto forwardedAddress =
        ForwardedObject::at(ref)->getForwardedAddress();
    std::cout << " to " << forwardedAddress;
    fixupSlot(context, slot, forwardedAddress);
  }
  std::cout << std::endl;
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
    walk<S>(context, target, visitor);
  }
};

/// Fix up all slots in an object
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

} // namespace omtalk::gc

#endif