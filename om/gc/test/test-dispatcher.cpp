#include <atomic>
#include <catch2/catch.hpp>
#include <functional>
#include <om/GC/Dispatcher.h>

using namespace om::gc;

TEST_CASE("Create worker"
          "[Worker]") {}

TEST_CASE("Startup 1 threads", "[Dispatcher]") {
  Dispatcher d(1);
  REQUIRE(d.getWorkerCount() == 1);
  d.waitForCompletion();
  REQUIRE(d.isAllWorkComplete() == true);
  d.waitForCompletion();
}

TEST_CASE("Startup 2 threads", "[Dispatcher]") {
  Dispatcher d(2);
  REQUIRE(d.getWorkerCount() == 2);
  d.waitForCompletion();
  REQUIRE(d.isAllWorkComplete() == true);
  d.waitForCompletion();
}

TEST_CASE("1 thread", "[Dispatcher]") {
  Dispatcher d(1);

  std::atomic<unsigned> i = 0;
  d.run([&i](WorkerContext &context) {
    REQUIRE(context.getID() == 0);
    i.store(1);
  });
  d.waitForCompletion();
  REQUIRE(i.load() == 1);
}

TEST_CASE("2 threads", "[Dispatcher]") {
  Dispatcher d(2);
  std::atomic<unsigned> i[2] = {0};
  d.run([&i](WorkerContext &context) {
    i[context.getID()] = context.getID() + 1;
  });
  d.waitForCompletion();
  REQUIRE(i[0] == 1);
  REQUIRE(i[1] == 2);
}
