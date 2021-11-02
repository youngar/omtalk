//===- OmOps.cpp - Omtalk dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
