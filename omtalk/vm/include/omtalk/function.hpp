#ifndef OMTALK_FUNCTION_HPP_
#define OMTALK_FUNCTION_HPP_

#include <cstdint>
#include <omtalk/vm/handle.hpp>

namespace omtalk {

// Denotes a stream of bytecodes
using Bytecodes = uint8_t *;

// A callable C function
using FnPtr = void (*)(int);

// A loaded function
struct MethodTableEntry {
  Bytecodes bytecodes;
  FnPtr function;
};

} // namespace omtalk

#endif // OMTALK_FUNCTION_HPP_