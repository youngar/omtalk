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

  template <typename VisitorT, typename... Args>
  void walk(VisitorT &visitor, Args... args) {
    metaLayout->walk(visitor, args...);
  }

  /// The layout of other layouts. A global singleton.
  gc::Ref<MetaLayout> metaLayout;
};

} // namespace om::om

#endif // OM_OM_GLOBAL_H
