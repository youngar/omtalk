//===- OmOps.cpp - Omtalk dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Om/IR/OmOps.h"
#include "mlir/Dialect/Om/IR/OmDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"
#include <mlir/IR/FunctionSupport.h>

using namespace mlir;
using namespace om;

namespace mlir {
namespace om {

#define GET_OP_CLASSES
#include "mlir/Dialect/Om/IR/Om.cpp.inc"

} // namespace om
} // namespace mlir
