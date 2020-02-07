#ifndef OMTALK_TEST_DIALECT_ENVIRONMENT_HPP_
#define OMTALK_TEST_DIALECT_ENVIRONMENT_HPP_

#include <gtest/gtest.h>

#include <omtalk/dialect.hpp>

namespace omtalk::test {

class DialectEnvironment : public ::testing::Environment {
  ~DialectEnvironment() override {}

  void SetUp() override {
       mlir::registerDialect<omtalk::Dialect>();
  }

  void TearDown() override {}
};

DialectEnvironment * dialect_env();

}  // namespace omtalk::test

#endif  // OMTALK_TEST_DIALECT_ENVIRONMENT_HPP_