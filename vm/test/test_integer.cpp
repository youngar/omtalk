#include <gtest/gtest.h>
#include <omtalk/omtalk.hpp>
#include <omtalk/stack.hpp>

using namespace omtalk;

TEST(Interpreter, stack) {
  Stack stack;

  std::uint8_t *sp = stack.data();
  std::uint8_t *bp = sp;

  EXPECT_NE(sp, nullptr);

  push_frame(sp, bp, 0);
  EXPECT_EQ(sp, bp);

  // push local
  push(sp, (vm::HeapPtr)1);
  EXPECT_EQ(*bp, 1);

  // push value
  push(sp, (vm::HeapPtr)2);
  EXPECT_EQ(*bp, 1);
  vm::HeapPtr local = get_local(sp, bp, 0);
  EXPECT_EQ(local, (vm::HeapPtr)1);

  // push arg
  push(sp, (vm::HeapPtr)5);

  // call a function
  push_frame(sp, bp, 1);
  push(sp, (vm::HeapPtr)3);
  vm::HeapPtr arg = get_arg(sp, bp, 0);
  EXPECT_EQ(arg, (vm::HeapPtr)5);
}

TEST(Integer, 1) {}