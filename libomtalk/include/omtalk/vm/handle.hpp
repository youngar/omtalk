#ifndef OMTALK_VM_HANDLE_HPP_
#define OMTALK_VM_HANDLE_HPP_

#include <cstddef>
#include <cstdint>

namespace omtalk {
namespace vm {

using HeapPtr = std::uint8_t*;

class Handle {
 public:
  HeapPtr get() const { return _ptr; }

  template <typename T = void>
  T* as_ptr() const {
    return reinterpret_cast<T*>(_ptr);
  }

 protected:
  Handle() : _ptr(nullptr) {}

  explicit Handle(HeapPtr ptr) : _ptr(ptr) {}

  Handle(const Handle&) = default;

  template <typename T>
  T* slot_ptr(std::size_t offset) const {
    return reinterpret_cast<T*>(_ptr + offset);
  }

  template <typename T>
  T& slot(std::size_t offset) const {
    return *slot_ptr<T>(offset);
  }

  template <typename T>
  T get_slot(std::size_t offset) const {
    return slot<T>(offset);
  }

  template <typename T>
  void set_slot(std::size_t offset, const T& value) const {
    slot<T>(offset) = value;
  }

  Handle& assign(HeapPtr ptr) {
    _ptr = ptr;
    return *this;
  }

  bool operator==(const Handle& rhs) {
    return this,get() == rhs.get();
  } 

 private:
  HeapPtr _ptr;
};

}  // namespace vm
}  // namespace omtalk

#endif  // OMTALK_VM_HANDLE_HPP_
