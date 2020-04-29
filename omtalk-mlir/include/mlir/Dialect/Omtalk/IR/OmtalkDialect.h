//===- OmtalkDialect.h - Omtalk dialect -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OMTALK_OMTALKDIALECT_H
#define MLIR_DIALECT_OMTALK_OMTALKDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace omtalk {

#include "mlir/Dialect/Omtalk/IR/OmtalkOpsDialect.h.inc"

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
enum Types {
  Box = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  BoxInt,
  BoxRef,
};
} // namespace OmtalkTypes

// class BoxType : public mlir::Type::TypeBase<BoxType, mlir::Type> {
// public:
//   using Base::Base;

//   static bool kindof(unsigned kind) { return kind == OmtalkTypes::Box; }

//   static BoxType get(mlir::MLIRContext *context) {
//     return Base::get(context, OmtalkTypes::Box);
//   }
// };

// class BoxIntType : public mlir::Type::TypeBase<BoxIntType, mlir::Type> {
// public:
//   using Base::Base;

//   static bool kindof(unsigned kind) { return kind == OmtalkTypes::BoxInt; }

//   static BoxIntType get(mlir::MLIRContext *context) {
//     return Base::get(context, OmtalkTypes::BoxInt);
//   }
// };

// class BoxRefType : public mlir::Type::TypeBase<BoxRefType, mlir::Type> {
// public:
//   using Base::Base;

//   static bool kindof(unsigned kind) { return kind == OmtalkTypes::BoxRef; }

//   static BoxRefType get(mlir::MLIRContext *context) {
//     return Base::get(context, OmtalkTypes::BoxRef);
//   }
// };

} // namespace omtalk
} // namespace mlir

#endif // MLIR_DIALECT_OMTALK_OMTALKDIALECT_H
