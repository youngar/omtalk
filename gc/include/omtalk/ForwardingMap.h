#ifndef OMTALK_FORWARDINGMAP_H
#define OMTALK_FORWARDINGMAP_H

#include <unordered_map>

namespace omtalk::gc {

using ForwardingMap = std::unordered_map<void *, void *>;

}

#endif