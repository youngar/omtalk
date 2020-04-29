//===- OmtalkOps.h - Omtalk dialect ops -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OMTALK_OMTALKOPS_H
#define MLIR_DIALECT_OMTALK_OMTALKOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffects.h"

namespace mlir {
namespace omtalk {

#define GET_OP_CLASSES
#include "mlir/Dialect/Omtalk/IR/OmtalkOps.h.inc"

} // namespace omtalk
} // namespace mlir

#endif // MLIR_DIALECT_OMTALK_OMTALKOPS_H
