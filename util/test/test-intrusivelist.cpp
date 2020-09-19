#include <catch2/catch.hpp>
#include <omtalk/Util/IntrusiveList.h>

using namespace omtalk;

struct Test;

using TestList = IntrusiveList<Test>;
using TestNode = TestList::Node;

struct Test {
  Test(unsigned value) : value(value) {}
  TestNode &getListNode() noexcept { return node; }
  unsigned value;
  TestNode node;
};

TEST_CASE("empty", "[IntrusiveList]") {
  TestList list;
  REQUIRE(list.empty() == true);
  REQUIRE(list.size() == 0);
}

TEST_CASE("push_front", "[IntrusiveList]") {
  TestList list;
  REQUIRE(list.empty() == true);
  REQUIRE(list.size() == 0);

  list.push_front(new Test(0));
  REQUIRE(list.empty() == false);
  REQUIRE(list.front().value == 0);
  REQUIRE(list.size() == 1);

  list.push_front(new Test(1));
  REQUIRE(list.empty() == false);
  REQUIRE(list.front().value == 1);
  REQUIRE(list.size() == 2);
}

TEST_CASE("front", "[IntrusiveList]") {
  TestList list;
  list.push_front(new Test(0));
  REQUIRE(list.front().value == 0);
  REQUIRE(list.back().value == 0);

  list.push_front(new Test(1));
  REQUIRE(list.front().value == 1);
  REQUIRE(list.back().value == 0);
}

TEST_CASE("splice", "[IntrusiveList]") {
  TestList a;
  TestList b;

  SECTION("splice empty lists") {
    a.splice(b);
    REQUIRE(a.empty());
    REQUIRE(b.empty());
  }

  SECTION("to list empty") {
    b.push_front(new Test(0));
    a.splice(b);
    REQUIRE(b.empty());
    REQUIRE(a.front().value == 0);
  }

  SECTION("from list empty") {
    a.push_front(new Test(0));
    a.splice(b);
    REQUIRE(b.empty());
    REQUIRE(a.front().value == 0);
  }

  SECTION("two full lists") {
    a.push_front(new Test(4));
    a.push_front(new Test(3));

    b.push_front(new Test(2));
    b.push_front(new Test(1));

    a.splice(b);

    REQUIRE(b.empty());
    REQUIRE(a.size() == 4);

    unsigned i = 1;
    for (const auto &t : a) {
      REQUIRE(t.value == i);
      i++;
    }
  }
}
