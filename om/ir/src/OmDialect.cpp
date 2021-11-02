#include "om/IR/OmDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "om/IR/OmOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace om;

void OmDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "om/IR/OmOps.cpp.inc"
      >();
}

#include "om/IR/OmDialect.cpp.inc"