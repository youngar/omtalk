//===- OmtalkDialect.cpp - Omtalk dialect ops -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "om/Dialect/Omtalk/IR/OmtalkDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "om/Dialect/Omtalk/IR/OmtalkOps.h"
#include "om/Support/LLVM.h"

using namespace mlir;
using namespace om;
using namespace om::omtalk;

//===----------------------------------------------------------------------===//
// Omtalk dialect.
//===----------------------------------------------------------------------===//

OmtalkDialect::OmtalkDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "om/Dialect/Omtalk/IR/Omtalk.cpp.inc"
      >();
  addTypes<BoxUnkType, BoxIntType, BoxRefType>();
}

//===----------------------------------------------------------------------===//
// Omtalk types.
//===----------------------------------------------------------------------===//

mlir::Type OmtalkDialect::parseType(mlir::DialectAsmParser &parser) const {
  if (parser.parseKeyword("box") || parser.parseLess())
    return mlir::Type();

  if (parser.parseKeyword("int") || parser.parseGreater())
    return BoxIntType::get(getContext());

  if (parser.parseKeyword("ref") || parser.parseGreater())
    return BoxRefType::get(getContext());

  if (parser.parseOptionalQuestion() || parser.parseGreater())
    return BoxUnkType::get(getContext());

  return mlir::Type();
}

void OmtalkDialect::printType(mlir::Type type,
                              mlir::DialectAsmPrinter &printer) const {
  // BoxType boxType = type.cast<BoxType>();
  if (type.isa<BoxUnkType>()) {
    printer << "box<?>";
  } else if (type.isa<BoxIntType>()) {
    printer << "box<int>";
  } else if (type.isa<BoxRefType>()) {
    printer << "box<ref>";
  } else {
    llvm_unreachable("Unhandled Linalg type");
  }
}
