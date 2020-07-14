//===- OmtalkOps.cpp - Omtalk dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Omtalk/IR/OmtalkOps.h"
#include "mlir/Dialect/Omtalk/IR/OmtalkDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"
#include <mlir/IR/FunctionSupport.h>

namespace mlir {
namespace omtalk {

static FunctionType createMethodType(OpBuilder &builder, unsigned args) {
  auto ty = BoxUnkType::get(builder.getContext());
  llvm::SmallVector<mlir::Type, 6> inputs(args, ty);
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

#define GET_OP_CLASSES
#include "mlir/Dialect/Omtalk/IR/OmtalkOps.cpp.inc"

} // namespace omtalk
} // namespace mlir
