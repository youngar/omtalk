//===- OmtalkOps.h - Omtalk dialect ops -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OMTALK_IR_OMTALKOPS_H
#define MLIR_DIALECT_OMTALK_IR_OMTALKOPS_H

#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <mlir/IR/FunctionSupport.h>
#include <mlir/IR/SymbolTable.h>

namespace mlir {
namespace omtalk {

#define GET_OP_CLASSES
#include "mlir/Dialect/Omtalk/IR/Omtalk.h.inc"

} // namespace omtalk
} // namespace mlir

#endif // MLIR_DIALECT_OMTALK_IR_OMTALKOPS_H
