#ifndef OMTALK_RUNTIME_H
#define OMTALK_RUNTIME_H

#include <om/GC/MemoryManager.h>
#include <omtalk/ObjectModel.h>

namespace omtalk {

//===----------------------------------------------------------------------===//
// Process
//===----------------------------------------------------------------------===//

class Process {};

//===----------------------------------------------------------------------===//
// Thread
//===----------------------------------------------------------------------===//

class Thread {
public:
  Thread(Process &proc) : proc(proc) {}

private:
  Process &proc;
};

//===----------------------------------------------------------------------===//
// Virtual Machine
//===----------------------------------------------------------------------===//

using MemoryManager = om::gc::MemoryManager<OmtalkCollectorScheme>;

struct VirtualMachineConfig {
  om::gc::MemoryManagerConfig memoryManagerConfig;
};

inline MemoryManager createMemoryManager(VirtualMachineConfig &vmConfig) {
  return om::gc::MemoryManagerBuilder<OmtalkCollectorScheme>()
      .withConfig(vmConfig.memoryManagerConfig)
      .build();
}

class VirtualMachine {
public:
  VirtualMachine(Thread &t, VirtualMachineConfig &vmConfig)
      : thread(t), memoryManager(createMemoryManager(vmConfig)) {
    // memoryManager.attach(thread.context);
  }

  MemoryManager &getMemoryManager() { return memoryManager; }

private:
  Thread &thread;
  MemoryManager memoryManager;
};

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//

using GCContext = om::gc::Context<OmtalkCollectorScheme>;

class Context {
public:
  Context(VirtualMachine &vm) : vm(vm), gcContext(vm.getMemoryManager()) {}
  VirtualMachine &vm;
  GCContext gcContext;
};

//===----------------------------------------------------------------------===//
// Bootstrapping
//===----------------------------------------------------------------------===//

/// Load important classes into the VM
bool bootstrap(VirtualMachine &vm);

} // namespace omtalk

#endif
