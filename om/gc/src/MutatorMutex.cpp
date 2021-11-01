#include <om/GC/MutatorMutex.h>
#include <type_traits>

using namespace om::gc;

static_assert(std::is_standard_layout_v<MutatorMutexData>);
