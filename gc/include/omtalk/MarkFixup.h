#ifndef OMTALK_MARKFIXUP_H
#define OMTALK_MARKFIXUP_H

#include <omtalk/Fixup.h>
#include <omtalk/Mark.h>
#include <omtalk/Record.h>
#include <omtalk/Ref.h>

namespace omtalk::gc {

template <typename S>
void mark(GlobalCollectorContext<S> &context, Ref<void> target) noexcept;

template <typename S>
class MarkFixupVisitor {
public:
  template <typename SlotProxyT>
  void visit(SlotProxyT slot, GlobalCollectorContext<S> &context) {
    fixupSlot<S>(context, slot);
    mark<S>(context, proxy::load<RELAXED>(slot));
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

} // namespace omtalk::gc

#endif