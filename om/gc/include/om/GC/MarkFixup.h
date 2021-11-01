#ifndef OM_GC_MARKFIXUP_H
#define OM_GC_MARKFIXUP_H

#include <om/GC/Fixup.h>
#include <om/GC/Mark.h>
#include <om/GC/Record.h>
#include <om/GC/Ref.h>

namespace om::gc {

template <typename S>
void mark(GlobalCollectorContext<S> &context, Ref<void> target) noexcept;

template <typename S>
class MarkFixupVisitor {
public:
  template <typename SlotProxyT>
  void visit(SlotProxyT slot, GlobalCollectorContext<S> &context) {
    fixupSlot<S>(context, slot);
    mark<S>(context, ab::proxy::load<ab::RELAXED>(slot));
  }
};

template <typename S>
struct MarkFixup {
  void operator()(GlobalCollectorContext<S> &context,
                  ObjectProxy<S> target) const noexcept {
    MarkFixupVisitor<S> visitor;
    record<S>(context, target);
    walk<S>(context, target, visitor);
  }
};

template <typename S>
void markFixup(GlobalCollectorContext<S> &context,
               ObjectProxy<S> target) noexcept {
  MarkFixup<S>()(context, target);
}

} // namespace om::gc

#endif