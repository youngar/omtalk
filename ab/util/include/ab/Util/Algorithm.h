#ifndef AB_UTIL_ALGORITHM_H
#define AB_UTIL_ALGORITHM_H

#include <algorithm>

namespace ab {

/// note: clone of llvm::copy
template <typename T, typename Iter>
Iter copy(T &&source, Iter destination) {
  return std::copy(std::begin(source), std::end(source), destination);
}

} // namespace ab

#endif
