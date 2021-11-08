#include <om/Om/Context.h>
#include <om/Om/MemoryManager.h>
#include <om/Om/StructLayout.h>
#include <om/Om/StructLayoutBuilder.h>

#include <catch2/catch.hpp>

TEST_CASE("Struct Layout", "[basic]") {
  auto memoryManager = om::om::makeMemoryManager();
  auto context = om::om::Context(memoryManager);

  auto b = om::om::StructLayoutBuilder();
  b.i32();
  b.i32();

  auto layout = b.build(context);

  REQUIRE(layout != nullptr);
}
