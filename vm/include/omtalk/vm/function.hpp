#ifndef OMTALK_VM_FUNCTION_HPP_
#define OMTALK_VM_FUNCTION_HPP_

#include <omtalk/vm/handle.hpp>
#include <omtalk/vm/klass.hpp>

namespace omtalk {
namespace vm {

struct FunctionField {
 public:
  static constexpr std::size_t KLASS = 0;
  static constexpr std::size_t BYTECODES = 8;
};

class FunctionHandle : Handle {
 public:
  FunctionHandle(HeapPtr ptr) : Handle(ptr) {}

  KlassHandle klass() const {
    return KlassHandle(get_slot<HeapPtr>(FunctionField::KLASS));
  }

  std::intptr_t bytecodes() const {
    return get_slot<std::intptr_t>(FunctionField::BYTECODES);
  }
};

}  // namespace vm
}  // namespace omtalk

#endif  // OMTALK_VM_FUNCTION_HPP_
