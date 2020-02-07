#ifndef OMTALK_VM_INTEGER_HPP_
#define OMTALK_VM_INTEGER_HPP_

#include <omtalk/vm/handle.hpp>
#include <omtalk/vm/klass.hpp>
#include <omtalk/vm/object.hpp>

namespace omtalk {
namespace vm {

constexpr std::size_t INTEGER_PTR_DATA_SIZE = 8;
constexpr std::size_t INTEGER_BIN_DATA_SIZE = 8;
constexpr std::size_t INTEGER_ALL_DATA_SIZE = 16;

class IntegerField {
 public:
  // Ptr Slots
  static constexpr std::size_t KLASS = 0;  // object

  // Bin Slots
  static constexpr std::size_t VALUE = 8;
};

class IntegerHandle : public Handle {
 public:
  IntegerHandle(HeapPtr ptr) : Handle(ptr) {}

  KlassHandle klass() const {
    return KlassHandle(get_slot<HeapPtr>(IntegerField::KLASS));
  }

  std::intptr_t value() const {
    return get_slot<std::intptr_t>(IntegerField::VALUE);
  }

  Handle& operator=(HeapPtr ptr) {
    _ptr = ptr;
    return *this;
  }
};

}  // namespace vm
}  // namespace omtalk

#endif  // OMTALK_VM_INTEGER_HPP_
