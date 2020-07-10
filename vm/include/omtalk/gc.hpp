#ifndef OMTALK_GC_HPP_
#define OMTALK_GC_HPP_

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omtalk/vm/handle.hpp>
#include <stdexcept>

namespace omtalk {

constexpr std::size_t HEAP_BYTE = 0x1;
constexpr std::size_t HEAP_MB = 0x100000;

constexpr std::size_t HEAP_ALIGNMENT = 8;
constexpr std::size_t HEAP_SLOT_SIZE = 8;
constexpr std::size_t HEAP_MINIMUM_SIZE = 16;

struct FreeEntry {};

struct MemoryOptions {
  // 4mb
  std::size_t heap_size = 0x400000;
};

class MemoryManagerException : public std::runtime_error {
 public:
  explicit MemoryManagerException(const std::string& what_arg);
  explicit MemoryManagerException(const char* what_arg);
};

inline MemoryManagerException::MemoryManagerException(
    const std::string& what_arg)
    : runtime_error(what_arg) {}

inline MemoryManagerException::MemoryManagerException(const char* what_arg)
    : runtime_error(what_arg) {}

class MemoryManager {
 public:
  MemoryManager();
  MemoryManager(MemoryOptions m);
  ~MemoryManager();
  vm::HeapPtr allocate_nogc(std::size_t size);
  vm::HeapPtr allocate_gc(std::size_t size);
  void collect();
  vm::HeapPtr heap_base() const { return _heap_base; }

 private:
  vm::HeapPtr _heap_base;
  vm::HeapPtr _heap_top;
  vm::HeapPtr _high_mark;
};

inline MemoryManager::MemoryManager()
    : MemoryManager::MemoryManager(MemoryOptions()) {}

inline MemoryManager::MemoryManager(MemoryOptions m) {
  _heap_base = reinterpret_cast<vm::HeapPtr>(malloc(m.heap_size));
  if (_heap_base == nullptr) {
    throw MemoryManagerException("Heap initialization failed");
  }
  _heap_top = _heap_base + m.heap_size;
  _high_mark = _heap_base;
}

inline MemoryManager::~MemoryManager() { free(_heap_base); }

inline vm::HeapPtr MemoryManager::allocate_nogc(std::size_t size) {
  std::cout << "Alloc " << size;
  if (_high_mark + size > _heap_top) {
    std::cout << " FAILED" << std::endl;
    throw MemoryManagerException("Out of memory");
  }
  vm::HeapPtr alloc = _high_mark;
  std::cout << " @ " << (void*)alloc << std::endl;
  _high_mark += size;
  return alloc;
}

inline vm::HeapPtr MemoryManager::allocate_gc(std::size_t slots) {
  try {
    return allocate_nogc(slots);
  } catch (MemoryManagerException& e) {
    collect();
    return allocate_nogc(slots);
  }
}

inline void MemoryManager::collect() {
  // no implementation
}

}  // namespace omtalk

#endif  // OMTALK_GC_HPP_