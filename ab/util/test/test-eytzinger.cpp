#include <ab/Util/Assert.h>
#include <ab/Util/Eytzinger.h>
#include <ab/Util/ValueIter.h>
#include <catch2/catch.hpp>
#include <iterator>
#include <utility>
#include <vector>

using namespace ab;

TEST_CASE("Default constructor", "[eytzinger]") {
  EytzingerTree<int, int> tree;
  REQUIRE(tree.size() == 0);
}

TEST_CASE("One element", "[eytzinger]") {
  EytzingerTree<int, int> tree(1);
  REQUIRE(tree.size() == 1);
}

TEST_CASE("Insertion and Set test", "[eytzinger]") {
  std::vector<int> values{1, 2, 3, 4, 5, 6, 7};
  std::vector<int> result{4, 2, 6, 1, 3, 5, 7};
  EytzingerTree<int, int> tree;
  tree.rebuild(values.begin(), values.end());

  for (std::size_t i = 0; i < values.size(); i++) {
    tree[values[i]] = values[i];
  }

  int i = 0;
  for (const auto &p : tree) {
    REQUIRE(p.first == result[i]);
    REQUIRE(p.first == p.second);
    ++i;
  }
}

TEST_CASE("Non-full binary tree", "[eytzinger]") {
  std::vector<int> values{1, 2, 3, 4, 5};
  std::vector<int> result{4, 2, 5, 1, 3};
  EytzingerTree<int, int> tree;
  tree.rebuild(values.begin(), values.end());

  for (std::size_t i = 0; i < values.size(); i++) {
    tree[values[i]] = values[i];
  }

  int i = 0;
  for (const auto &p : tree) {
    REQUIRE(p.first == result[i]);
    REQUIRE(p.first == p.second);
    ++i;
  }
}

TEST_CASE("eytzinger2", "[eytzinger]") {
  auto limit = 128;
  for (auto n = 0; n < limit; ++n) {
    EytzingerTree<int, int> tree(n);
    tree.rebuild(toIter(0), toIter(n));
    for (auto i = 0; i < n; ++i) {
      REQUIRE(tree[i] == 0);
    }
    for (auto i = 0; i < n; ++i) {
      tree[i] = i;
    }
    for (int i = 0; i < n; ++i) {
      REQUIRE(tree[i] == i);
    }
  }
}
