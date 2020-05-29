#ifndef OMTALK_VM_OBJECT_HPP_
#define OMTALK_VM_OBJECT_HPP_

#include <omtalk/vm/handle.hpp>
#include <omtalk/vm/klass.hpp>

namespace omtalk {
namespace vm {

constexpr std::size_t OBJECT_PTR_DATA_SIZE = 8;
constexpr std::size_t OBJECT_BIN_DATA_SIZE = 0;
constexpr std::size_t OBJECT_ALL_DATA_SIZE = 8;

struct ObjectField {
  static constexpr std::size_t KLASS = 0;
};

class ObjectHandle : Handle {
 public:
  explicit ObjectHandle(HeapPtr ptr) : Handle(ptr) {}

  KlassHandle klass() const {
    return KlassHandle(get_slot<HeapPtr>(ObjectField::KLASS));
  }
};

}  // namespace vm
}  // namespace omtalk

#endif  // OMTALK_VM_OBJECT_HPP_
