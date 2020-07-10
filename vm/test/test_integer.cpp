#include <catch2/catch.hpp>
#include <omtalk/omtalk.hpp>
#include <omtalk/stack.hpp>

using namespace omtalk;

TEST_CASE(Interpreter, stack) {
  Stack stack;

  std::uint8_t *sp = stack.data();
  std::uint8_t *bp = sp;

  REQUIRE(sp == nullptr);

  push_frame(sp, bp, 0);
  (sp, bp);

  // push local
  push(sp, (vm::HeapPtr)1);
  REQUIRE(*bp == 1);

  // push value
  push(sp, (vm::HeapPtr)2);
  REQUIRE(*bp == 1);
  vm::HeapPtr local = get_local(sp, bp, 0);
  REQUIRE(local == (vm::HeapPtr)1);

  // push arg
  push(sp, (vm::HeapPtr)5);

  // call a function
  push_frame(sp, bp, 1);
  push(sp, (vm::HeapPtr)3);
  vm::HeapPtr arg = get_arg(sp, bp, 0);
  REQUIRE(arg == (vm::HeapPtr)5);
}
