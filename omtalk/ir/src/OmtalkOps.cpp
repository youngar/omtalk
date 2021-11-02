#include "omtalk/IR/OmtalkOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/Transforms/InliningUtils.h"
#include "omtalk/IR/OmtalkDialect.h"
#include "omtalk/IR/OmtalkTypes.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace omtalk;

static FunctionType createMethodType(OpBuilder &builder, unsigned nargs) {
  auto ty = BoxUnkType::get(builder.getContext());
  llvm::SmallVector<mlir::Type, 6> inputs(nargs, ty);
  llvm::SmallVector<mlir::Type, 1> results(1, ty);
  return builder.getFunctionType(inputs, results);
}

void MethodOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     unsigned argCount) {

  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getTypeAttrName(),
                      TypeAttr::get(createMethodType(builder, argCount)));
  result.addRegion();
}

// ODS Op Implementations.
#define GET_OP_CLASSES
#include "omtalk/IR/OmtalkOps.cpp.inc"
