#include <gtest/gtest.h>

#include <omtalk/symbol.hpp>

using namespace omtalk;

TEST(symbol_table, empty) {
  SymbolTable t;
  Symbol s = -1;
  EXPECT_FALSE(t.contains("Test"));
  EXPECT_FALSE(t.contains(invalid_symbol));
}

TEST(symbol_table, insert) {
  Symbol s;
  SymbolTable t;

  s = t.intern("Test");
  EXPECT_NE(s, invalid_symbol);
  EXPECT_TRUE(t.contains("Test"));
  EXPECT_TRUE(t.contains(s));
  EXPECT_EQ(t.contains("Test"), s);

  Symbol s2 = t.intern("Test2");
  EXPECT_NE(s2, invalid_symbol);
  EXPECT_TRUE(t.contains("Test2"));
  EXPECT_TRUE(t.contains(s2));
  EXPECT_EQ(t["Test2"], s2);

  EXPECT_NE(s, s2);
}

TEST(string_table, insert) {
  StringTable t;
  std::string test = "Test";
  const char *result = t["Test"];

  EXPECT_STREQ(test.c_str(), result);
  EXPECT_STREQ(t["Test"], result);
  EXPECT_EQ(t["Test"], result);
  EXPECT_STREQ(t[test], result);
  EXPECT_EQ(t[test], result);

  result = t[std::move(std::string("Test2"))];
  EXPECT_STREQ(result, "Test2");

  const char *test3 = "Test3";
  EXPECT_STREQ(t[test3], test3);
  EXPECT_NE(t[test3], test3);
  EXPECT_EQ(t[test3], t["Test3"]);
}
