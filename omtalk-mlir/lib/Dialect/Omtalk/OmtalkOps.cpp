//===- StandaloneOps.cpp - Standalone dialect ops ---------------*- C++ -*-===//
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

namespace mlir {
namespace omtalk {

//===----------------------------------------------------------------------===//
// SendOp
//===----------------------------------------------------------------------===//

// mlir::CallInterfaceCallable SendOp::getCallableForCallee() {
//   return getAttrOfType<SymbolRefAttr>("message");
// }

// mlir::Operation::operand_range SendOp::getArgOperands() { return inputs(); }

//===----------------------------------------------------------------------===//
// SendIntAddOp
//===----------------------------------------------------------------------===//

// mlir::CallInterfaceCallable SendIntAddOp::getCallableForCallee() {
//   return getAttrOfType<SymbolRefAttr>("message");
// }

// mlir::Operation::operand_range SendIntAddOp::getArgOperands() {
//   return inputs();
// }

// OpFoldResult SendIntAddOp::fold(ArrayRef<Attribute> operands) {
/// addi(x, 0) -> x
// return {};
//   if (matchPattern(rhs(), m_Zero())) return lhs();

//   return constFoldBinaryOp<IntegerAttr>(operands,
//                                         [](APInt a, APInt b) { return a + b;
//                                         });
// }

//===----------------------------------------------------------------------===//
// SendRef
//===----------------------------------------------------------------------===//

// mlir::CallInterfaceCallable SendRefOp::getCallableForCallee() {
//   return getAttrOfType<SymbolRefAttr>("message");
// }

// mlir::Operation::operand_range SendRefOp::getArgOperands() { return inputs();
// }

//===----------------------------------------------------------------------===//
// ODS Ops
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Omtalk/IR/OmtalkOps.cpp.inc"

} // namespace omtalk
} // namespace mlir
