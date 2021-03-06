//===- OmtalkDialect.h - Omtalk dialect -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OMTALK_IR_OMTALKDIALECT_H
#define MLIR_DIALECT_OMTALK_IR_OMTALKDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace omtalk {

#include "mlir/Dialect/Omtalk/IR/OmtalkDialect.h.inc"

///
/// Omtalk Dialect
///

// mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

// void printType(mlir::Type type,
//                mlir::DialectAsmPrinter &printer) const override;

///
/// Omtalk Types
///

namespace OmtalkTypes {
enum Kinds {
  BoxUnk = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_2_TYPE,
  BoxInt,
  BoxRef,
};
} // namespace OmtalkTypes

class BoxUnkType : public mlir::Type::TypeBase<BoxUnkType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == OmtalkTypes::BoxUnk; }

  static BoxUnkType get(mlir::MLIRContext *context) {
    return Base::get(context, OmtalkTypes::BoxUnk);
  }
};

class BoxIntType : public mlir::Type::TypeBase<BoxIntType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == OmtalkTypes::BoxInt; }

  static BoxIntType get(mlir::MLIRContext *context) {
    return Base::get(context, OmtalkTypes::BoxInt);
  }
};

class BoxRefType : public mlir::Type::TypeBase<BoxRefType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == OmtalkTypes::BoxRef; }

  static BoxRefType get(mlir::MLIRContext *context) {
    return Base::get(context, OmtalkTypes::BoxRef);
  }
};

} // namespace omtalk
} // namespace mlir

#endif // MLIR_DIALECT_OMTALK_IR_OMTALKDIALECT_H
