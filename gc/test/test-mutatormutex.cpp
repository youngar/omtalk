#include <catch2/catch.hpp>
#include <omtalk/MutatorMutex.h>
#include <thread>

using namespace omtalk::gc;

TEST_CASE("MutatorMutex no request", "[garbage collector]") {
  MutatorMutex m;
  REQUIRE(m.requested() == false);
  m.lockShared();
  REQUIRE(m.requested() == false);
  m.unlockShared();
  REQUIRE(m.requested() == false);
}

TEST_CASE("MutatorMutex request", "[garbage collector]") {
  MutatorMutex m;
  m.lock();
  std::thread thread([&m] {
    m.lockShared();
    m.unlockShared();
  });
  m.unlock();
  thread.join();
}

// TEST_CASE("MutatorLock requested", "[garbage collector]") {
//   MutatorMutex m;
//   MutatorLock l(m);
//   REQUIRE(m.requested());
// }

// TEST_CASE("MutatorLock request", "[garbage collector]") {
//   MutatorMutex m;
//   MutatorLock l(m);
//   std::thread thread([&m] {
//     m.lockShared();
//     m.unlockShared();
//   });
//   l.unlock();
//   thread.join();
// }
