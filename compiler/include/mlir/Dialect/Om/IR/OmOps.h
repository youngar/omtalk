//===- OmOps.h - Om dialect ops ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OM_IR_OMOPS_H
#define MLIR_DIALECT_OM_IR_OMOPS_H

#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <mlir/IR/FunctionSupport.h>
#include <mlir/IR/SymbolTable.h>

namespace mlir {
namespace om {

#define GET_OP_CLASSES
#include "mlir/Dialect/Om/IR/Om.h.inc"

} // namespace om
} // namespace mlir

#endif // MLIR_DIALECT_OM_IR_OMOPS_H
