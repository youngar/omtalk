#include <omtalk/MutatorMutex.h>
#include <type_traits>

using namespace omtalk::gc;

static_assert(std::is_standard_layout_v<MutatorMutexData>);
