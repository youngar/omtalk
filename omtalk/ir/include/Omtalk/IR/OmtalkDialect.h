//===- OmtalkDialect.h - Omtalk dialect -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OMTALK_IR_OMTALKDIALECT_H
#define MLIR_DIALECT_OMTALK_IR_OMTALKDIALECT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "omtalk/IR/LLVM.h"

#include "omtalk/IR/OmtalkDialect.h.inc"

///
/// Omtalk Dialect
///

// mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

// void printType(mlir::Type type,
//                mlir::DialectAsmPrinter &printer) const override;

///
/// Omtalk Types
///

namespace omtalk {

class BoxUnkType
    : public mlir::Type::TypeBase<BoxUnkType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
};

class BoxIntType
    : public mlir::Type::TypeBase<BoxIntType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
};

class BoxRefType
    : public mlir::Type::TypeBase<BoxRefType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
};

} // namespace omtalk

#endif // MLIR_DIALECT_OMTALK_IR_OMTALKDIALECT_H
