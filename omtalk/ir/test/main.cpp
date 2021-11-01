
#include <gtest/gtest.h>

#include "dialect_environment.h"

using namespace omtalk;
using namespace omtalk::test;

DialectEnvironment *dialect_env = nullptr;

DialectEnvironment *omtalk::test::dialect_env() { return ::dialect_env; }

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  ::dialect_env = new DialectEnvironment;
  ::testing::AddGlobalTestEnvironment(::dialect_env);

  return RUN_ALL_TESTS();
}
