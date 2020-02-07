#ifndef OMTALK_VM_KLASS_HPP_
#define OMTALK_VM_KLASS_HPP_

#include <omtalk/vm/handle.hpp>
#include <omtalk/symbol.hpp>

#include <unordered_map>

namespace omtalk {
namespace vm {

struct KlassData {
  std::map<Symbol, HeapPtr> methods;
};

constexpr std::size_t KLASS_PTR_DATA_SIZE =  8;
constexpr std::size_t KLASS_BIN_DATA_SIZE =  8;
constexpr std::size_t KLASS_ALL_DATA_SIZE = 16;

struct KlassField {
  static constexpr std::size_t KLASS = 0;
  static constexpr std::size_t DATA = 8;
};

class KlassHandle : public Handle {
 public:
  KlassHandle() = default;

  explicit KlassHandle(HeapPtr ptr) : Handle(ptr) {}

  KlassHandle(const KlassHandle&) = default;

  KlassHandle klass() const {
    return KlassHandle(get_slot<HeapPtr>(KlassField::KLASS));
  }

  void set_klass(HeapPtr klass) const {
    set_slot<HeapPtr>(KlassField::KLASS, klass);
  }

  KlassData* data() const {
    return get_slot<KlassData*>(KlassField::DATA);
  }
};

}  // namespace vm
}  // namespace omtalk

#endif  // OMTALK_VM_KLASS_HPP_
