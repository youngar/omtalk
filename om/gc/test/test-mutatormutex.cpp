#include <catch2/catch.hpp>
#include <om/GC/MutatorMutex.h>
#include <thread>

using namespace om::gc;

TEST_CASE("MutatorMutex no request", "[garbage collector]") {

  MutatorMutex m;
  REQUIRE(m.requested() == false);
  m.attach();
  REQUIRE(m.requested() == false);
  m.detach();
  REQUIRE(m.requested() == false);
}

TEST_CASE("MutatorMutex request", "[garbage collector]") {
  MutatorMutex m;
  m.lock();
  std::thread thread([&m] {
    m.attach();
    m.detach();
  });
  m.unlock();
  thread.join();
}

TEST_CASE("MutatorLock requested", "[garbage collector]") {
  MutatorMutex m;
  MutatorLock l(m);
  REQUIRE(m.requested());
}

TEST_CASE("MutatorLock request", "[garbage collector]") {
  MutatorMutex m;
  MutatorLock l(m);
  std::thread thread([&m] {
    m.attach();
    m.detach();
  });
  l.unlock();
  thread.join();
}
