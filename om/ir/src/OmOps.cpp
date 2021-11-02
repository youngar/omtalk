#include "om/IR/OmOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/Transforms/InliningUtils.h"
#include "om/IR/OmDialect.h"

using namespace mlir;
using namespace om;

// ODS Op Implementations.
#define GET_OP_CLASSES
#include "om/IR/OmOps.cpp.inc"
