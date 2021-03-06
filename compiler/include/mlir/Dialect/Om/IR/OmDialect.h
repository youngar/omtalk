//===- OmDialect.h - Om dialect ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OM_IR_OMDIALECT_H
#define MLIR_DIALECT_OM_IR_OMDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace om {

#include "mlir/Dialect/Om/IR/OmDialect.h.inc"

} // namespace om
} // namespace mlir

#endif // MLIR_DIALECT_OM_IR_OMDIALECT_H
