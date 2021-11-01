#ifndef OMTALK_INTERPRETER_HPP_
#define OMTALK_INTERPRETER_HPP_

#include <cstdint>
#include <omtalk/vm/handle.hpp>
#include <omtalk/vmstructs.h>

namespace omtalk {

// Stack

inline void push(std::uint8_t *&sp, vm::HeapPtr value) {
  vm::HeapPtr *slot = (vm::HeapPtr *)sp;
  *slot = value;
  sp += 8;
}

inline vm::HeapPtr pop(std::uint8_t *&sp) {
  vm::HeapPtr *slot = (vm::HeapPtr *)sp;
  vm::HeapPtr value = *slot;
  sp -= 8;
  return value;
}

inline void push_frame(std::uint8_t *&sp, std::uint8_t *&bp,
                       std::uintptr_t nlocals) {
  push(sp, bp);
  bp = sp;
  sp += (8 * nlocals);
}

inline void pop_frame(std::uint8_t *&sp, std::uint8_t *&bp) {
  sp = bp;
  bp = pop(sp);
}

inline vm::HeapPtr get_local(std::uint8_t *&sp, std::uint8_t *&bp,
                             std::uintptr_t n) {
  vm::HeapPtr *slot = (vm::HeapPtr *)bp;
  return slot[n];
}

inline void push_local(std::uint8_t *&sp, std::uint8_t *&bp, std::uintptr_t n) {
  push(sp, get_local(sp, bp, n));
}

inline char load_bc(std::uint8_t *pc) { return *reinterpret_cast<char *>(pc); }

inline vm::HeapPtr get_arg(std::uint8_t *&sp, std::uint8_t *&bp,
                           std::uintptr_t n) {
  vm::HeapPtr *slot = (vm::HeapPtr *)bp;
  return slot[-1 - n];
}

inline void push_arg(std::uint8_t *&sp, std::uint8_t *&bp, std::uintptr_t n) {
  push(sp, get_arg(sp, bp, n));
}

// interpreter

extern "C" void omtalk_interpret(OmtalkThread &thread);

} // namespace omtalk

#endif // OMTALK_INTERPRETER_HPP_
