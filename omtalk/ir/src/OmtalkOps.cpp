//===- OmtalkOps.cpp - Omtalk dialect ops -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "omtalk/IR/OmtalkOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/Transforms/InliningUtils.h"
#include "omtalk/IR/OmtalkDialect.h"
#include "om/Support/LLVM.h"

using namespace mlir;
using namespace om;
using namespace om::omtalk;

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

#define GET_OP_CLASSES
#include "mlir/Dialect/Omtalk/IR/Omtalk.cpp.inc"
