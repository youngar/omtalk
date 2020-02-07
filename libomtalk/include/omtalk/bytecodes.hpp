#ifndef OMTALK_BYTECODES_HPP_
#define OMTALK_BYTECODES_HPP_

#include <cstdlib>

namespace omtalk {

// enum class Bytecode {
// HALT             =  0,
// DUP              =  1,
// PUSH_LOCAL       =  2,
// PUSH_ARGUMENT    =  3,
// PUSH_FIELD       =  4,
// PUSH_BLOCK       =  5,
// PUSH_CONSTANT    =  6,
// PUSH_GLOBAL      =  7,
// POP              =  8,
// POP_LOCAL        =  9,
// POP_ARGUMENT     = 10,
// POP_FIELD        = 11,
// SEND             = 12,
// SUPER_SEND       = 13,
// RETURN_LOCAL     = 14,
// RETURN_NON_LOCAL = 15
// ADD              = 16
// MULTIPLY         = 17
// SUBTRACT         = 18
// };

enum Bytecode { HALT, NOP, RETURN, PUSH_CONST, PUSH_GLOBAL, SEND };


constexpr std::size_t HALT_SIZE = 1;
constexpr std::size_t NOP_SIZE = 1;
constexpr std::size_t RETURN_SIZE = 1;
constexpr std::size_t PUSH_CONST_SIZE = 9;
constexpr std::size_t PUSH_GLOBAL_SIZE = 9;
constexpr std::size_t SEND_SIZE = 9;


}  // namespace omtalk

#endif  // OMTALK_BYTECODES_HPP_
