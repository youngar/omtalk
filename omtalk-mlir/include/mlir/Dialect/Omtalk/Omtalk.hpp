#ifndef OMTALK_OMTALK_HPP_
#define OMTALK_OMTALK_HPP_

#include <cstdint>

namespace omtalk {

///
/// Base Classes
///

struct Klass;

struct Object {
  Klass *klass;
};

struct Klass {
  Klass *klass;
};

struct Integer {
  Klass *klass;
  std::int64_t value;
};

///
/// VirtualMachine
///

// Process

class Process {};

// Thread

class Thread {
 public:
  Thread(Process &proc) : _proc(proc) {}

 private:
  Process &_proc;
};

class VirtualMachine {
 public:
  VirtualMachine(Thread &t);

  Klass *klass_class;
  Klass *klass_integer;
  Klass *klass_object;

 private:
  bool load_classes();
  Thread &_thread;
};

inline VirtualMachine::VirtualMachine(Thread &t) : _thread(t) {
  load_classes();
}

inline bool VirtualMachine::load_classes() {
  klass_class = new Klass{};
  klass_class->klass = klass_class;

  klass_object = new Klass{};
  klass_object->klass = klass_class;

  klass_integer = new Klass();
  klass_integer->klass = klass_class;

  return true;
}

}  // namespace omtalk

#endif  // OMTALK_OMTALK_HPP_