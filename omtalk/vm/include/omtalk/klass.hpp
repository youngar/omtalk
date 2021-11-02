#ifndef OMTALK_KLASS_HPP_
#define OMTALK_KLASS_HPP_

#include <cstdint>
#include <map>
#include <omtalk/bytecodes.hpp>
#include <omtalk/vm/klass.hpp>
#include <omtalk/vm/symbol.hpp>
#include <vector>

namespace omtalk {

// using Bytecode = std::uint8_t;

class String {};

// class Symbol {
//  private:
//   vm::SymbolHandle _symbol_handle;
//   std::string _symbol;
// };

struct Klass {
  Symbol name;
  vm::KlassHandle _this;
  Klass *super_klass = nullptr;
  // std::vector<MethodDef> _methods;
};

class Fields {};

class Methods {};

class ConstantPoolEntry {};

class ConstantPool {
  std::vector<ConstantPoolEntry> _constant_pool;
};

enum class CPItemType {
  UNUSED,
  SYMBOL,
  STRING,
  METHOD,
  FIELD,
  CLASS,
};

enum SendTarget {
  SEND_GENERIC,
  SEND_INTEGER_ADD,
  SEND_INTEGER_SUBTRACT,
};

class MethodDef {
  Symbol selector;
  SendTarget send_target;
  std::vector<std::uint8_t> bytecode;
  void *jit_address;
};

} // namespace omtalk

#endif // OMTALK_KLASS_HPP_