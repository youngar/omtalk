#ifndef OM_PVALUE_HPP_
#define OM_PVALUE_HPP_

namespace om {

class PValueProxy {
public:
};

enum class  { REF, INT };

struct AsRef {};
struct AsInt {};

constexpr AsRef AS_REF;
constexpr AsInt AS_INT;

/// Untyped tagged value. T stands for "tagged".
class TVal {
public:
  PValue(AsRef, void *value) : as_ptr(value) {}

  template <ty private : union {
    void *as_ptr;
    std::uintptr_t as_uint;
    std::intptr_t as_int;
  };

  isSmi() 
};

} // namespace om

#endif // OM