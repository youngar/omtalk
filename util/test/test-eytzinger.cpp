#include <catch2/catch.hpp>
#include <iterator>
#include <omtalk/Util/Assert.h>
#include <omtalk/Util/Eytzinger.h>
#include <utility>
#include <vector>

using namespace omtalk;

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

  for (int i = 0; i < values.size(); i++) {
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

  for (int i = 0; i < values.size(); i++) {
    tree[values[i]] = values[i];
  }

  int i = 0;
  for (const auto &p : tree) {
    REQUIRE(p.first == result[i]);
    REQUIRE(p.first == p.second);
    ++i;
  }
}

template <typename T>
class ValueIter {
public:
  using difference_type = T;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::forward_iterator_tag;

  explicit ValueIter(T x) noexcept : x(x) {}

  ValueIter &operator++() noexcept {
    ++x;
    return *this;
  }

  bool operator!=(const ValueIter &rhs) const noexcept { return x != rhs.x; }

  T operator*() const noexcept { return x; }

  T operator-(const ValueIter &rhs) const noexcept { return x - rhs.x; }

private:
  T x;
};

template <typename T>
auto toIter(T x) -> ValueIter<T> {
  return ValueIter<T>(x);
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
    for (auto i = 0ul; i < n; ++i) {
      REQUIRE(tree[i] == i);
    }
  }
}
