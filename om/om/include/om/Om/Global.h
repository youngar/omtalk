#ifndef OM_OM_GLOBAL_H

#include <om/GC/Ref.h>
#include <om/Om/Context.h>

namespace om::om {

struct MetaLayout;

class Global {
public:
  Global();

  void init(Context &context) noexcept;

  gc::Ref<MetaLayout> loadMetaLayout(Context &context) noexcept;

  template <typename ContextT, typename VisitorT>
  void walk(ContextT &context, VisitorT &visitor) {
    visitor.visit(context, gc::RefProxy(&metaLayout));
  }

  gc::Ref<MetaLayout> metaLayout;
};

} // namespace om::om

#endif // OM_OM_GLOBAL_H
