#include <catch2/catch.hpp>
#include <omtalk/ArenaManager.h>

namespace ot = omtalk;
namespace gc = ot::gc;

TEST_CASE("", "[arena map]") {
  gc::ArenaMap map;
  REQUIRE(!map.managed(nullptr));
  int x;
  REQUIRE(!map.managed(gc::arenaPtr(&x)));
}

TEST_CASE("Not Managed", "[arena manager]") {
  gc::ArenaManager manager;

  REQUIRE(!manager.managed(nullptr));

  int x;
  REQUIRE(!manager.managed(gc::arenaPtr(&x)));
}

TEST_CASE("Allocate Arena", "[arena manager]") {
  gc::ArenaManager manager;

  auto arena = reinterpret_cast<char *>(manager.allocate());
  REQUIRE(arena != nullptr);
  REQUIRE(ot::aligned(arena, gc::ARENA_SIZE));
  REQUIRE(manager.managed(arena));
  REQUIRE(manager.managed(arena + 1));
  REQUIRE(manager.managed(arena + gc::ARENA_SIZE / 2));
  REQUIRE(manager.managed(arena + gc::ARENA_SIZE - 1));
  REQUIRE(!manager.managed(arena + gc::ARENA_SIZE));

  manager.free(arena);
  REQUIRE(!manager.managed(arena));
  REQUIRE(!manager.managed(arena + gc::ARENA_SIZE));

  auto arena2 = manager.allocate();
  REQUIRE(arena == arena2);
  manager.free(arena2);
}

TEST_CASE("Allocate Two Arenas", "[arena manager]") {
  gc::ArenaManager manager;

  auto arena1 = reinterpret_cast<char *>(manager.allocate());
  auto arena2 = reinterpret_cast<char *>(manager.allocate());

  REQUIRE(arena1 != nullptr);
  REQUIRE(arena2 != nullptr);

  REQUIRE(ot::aligned(arena1, gc::ARENA_SIZE));
  REQUIRE(ot::aligned(arena2, gc::ARENA_SIZE));

  REQUIRE(manager.managed(arena1));
  REQUIRE(manager.managed(arena2));

  manager.free(arena1);
  REQUIRE(!manager.managed(arena1));
  REQUIRE(manager.managed(arena2));

  manager.free(arena2);
  REQUIRE(!manager.managed(arena1));
  REQUIRE(!manager.managed(arena2));
}