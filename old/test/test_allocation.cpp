#include <gtest/gtest.h>

#include <omtalk/gc.hpp>
#include <omtalk/omtalk.hpp>
#include <omtalk/vm/object.hpp>

using namespace omtalk;

TEST(Allocation, allocate) {
  MemoryOptions options;
  options.heap_size = 0x8;

  MemoryManager mm(options);

  vm::HeapPtr o;
  EXPECT_NO_THROW(o = mm.allocate_nogc(0x8));
  EXPECT_NE(o, nullptr);
  EXPECT_ANY_THROW(o = mm.allocate_nogc(0x8));
  EXPECT_ANY_THROW(o = mm.allocate_gc(0x8));
}

TEST(Allocation, initialization) {
  omtalk::Process process;
  omtalk::Thread thread(process);
  omtalk::VirtualMachine vm(thread);
}


