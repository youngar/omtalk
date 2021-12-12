#ifndef OM_OBJECTPROXY_H_
#define OM_OBJECTPROXY_H_

#include <om/Scheme.h>
#include <omtalk/MemoryManager.h>

namespace om {

class ObjectProxy {
public:
  template <typename C, typename V>
  void walk(C &c, V &v) const noexcept {
    switch (target->kind) {
    case 
    }
    }
};

}

#endif // OM_OBJECTPROXY_H_
