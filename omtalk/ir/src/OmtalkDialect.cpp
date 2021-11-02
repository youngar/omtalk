#include "omtalk/IR/OmtalkDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "omtalk/IR/OmtalkOps.h"

using namespace mlir;
using namespace omtalk;

void OmtalkDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "omtalk/IR/OmtalkOps.cpp.inc"
      >();

  // Register types.
  registerTypes();
}

// ODS Dialect implementation.
#include "omtalk/IR/OmtalkDialect.cpp.inc"