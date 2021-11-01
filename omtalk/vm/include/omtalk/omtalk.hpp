#ifndef OMTALK_OMTALK_HPP_
#define OMTALK_OMTALK_HPP_

#include <omtalk/gc.hpp>
#include <omtalk/interpreter.hpp>
#include <omtalk/omtalk.hpp>
#include <omtalk/stack.hpp>
#include <omtalk/vm/integer.hpp>
#include <omtalk/vm/klass.hpp>

namespace omtalk {

// Process

class Process {};

// Thread

class Thread {
public:
  Thread(Process &proc) : _proc(proc), _stack(100) {}

private:
  Process &_proc;
  Stack _stack;
};

class VirtualMachine {
public:
  VirtualMachine(Thread &t);

private:
  vm::KlassHandle allocate_klass();

  vm::KlassHandle new_klass();

  vm::IntegerHandle new_integer();

  bool load_classes();
  void bootstrap();
  Thread &_thread;
  MemoryManager mm;

  SymbolTable _symbol_table;

  vm::KlassHandle k_function;
  vm::KlassHandle k_integer;
  vm::KlassHandle k_klass;
  vm::KlassHandle k_object;
};

inline vm::KlassHandle VirtualMachine::allocate_klass() {
  return vm::KlassHandle(mm.allocate_nogc(vm::KLASS_ALL_DATA_SIZE));
}

inline vm::KlassHandle VirtualMachine::new_klass() {
  vm::KlassHandle klass = allocate_klass();
  klass.set_klass(k_klass.get());
  return klass;
}

inline VirtualMachine::VirtualMachine(Thread &t) : _thread(t), {
  load_classes();
  bootstrap();
}

inline bool VirtualMachine::load_classes() { return true; }

inline void VirtualMachine::bootstrap() {

  k_klass = allocate_klass();
  k_klass.set_klass(k_klass.get());

  k_object = allocate_klass();
  k_object.set_klass(k_klass.get());

  k_string = allocate_klass();
  k_string.set_klass(k_klass.get());

  k_symbol = allocate_klass();
  k_symbol.set_klass(k_klass.get());
}

} // namespace omtalk

#endif // OMTALK_OMTALK_HPP_