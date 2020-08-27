#include <catch2/catch.hpp>
#include <omtalk/ArenaManager.h>

using namespace omtalk;

TEST_CASE("Not Managed", "[arena manager]") {
  ArenaManager manager;
  
  REQUIRE(!manager.managed(nullptr));
  REQUIRE(!manager.managed(reinterpret_cast<void *>(1)));

  int x;
  REQUIRE(!manager.managed(&x));
}

TEST_CASE("Allocate Arena", "[arena manager]") {
  ArenaManager manager;
  
  auto arena = reinterpret_cast<char *>(manager.allocate());
  REQUIRE(arena != nullptr);
  REQUIRE(aligned(arena, ARENA_SIZE));
  REQUIRE(manager.managed(arena));
  REQUIRE(manager.managed(arena + 1));
  REQUIRE(manager.managed(arena + ARENA_SIZE / 2));
  REQUIRE(manager.managed(arena + ARENA_SIZE - 1));
  REQUIRE(!manager.manged(arena + ARENA_SIZE));

  manager.free(arena);
  REQUIRE(!manager.managed(arena));
  REQUORE(!manager.managed(arena + ARENA_SIZE));

  auto arena2 = manager.allocate();
  REQUIRE(arena == arena2);
  manager.free(arena2);
}

TEST_CASE("Allocate Two Arenas", "[arena manager]") {
  ArenaManager manager;
  
  auto arena1 = reinterpret_cast<char *>(manager.allocate());
  auto arena2 = reinterpret_cast<char *>(manager.allocate());

  REQUIRE(arena1 != nullptr);
  REQUIRE(arena2 != nullptr);

  REQUIRE(aligned(arena1, ARENA_SIZE));
  REQUIRE(aligned(arena2, ARENA_SIZE));

  REQUIRE(manager.managed(arena1));
  REQUIRE(manager.managed(arena2));

  manager.free(arena);
  REQUIRE(!manager.managed(arena));
  REQUORE(!manager.managed(arena + ARENA_SIZE));

  auto arena2 = manager.allocate();
  REQUIRE(arena == arena2);
  manager.free(arena2);
}