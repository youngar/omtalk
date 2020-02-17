#ifndef OMTALK_DIALECT_HPP_
#define OMTALK_DIALECT_HPP_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

///
/// Omtalk Opersations
///

namespace mlir {
namespace omtalk {

#define GET_OP_CLASSES
#include <omtalk/ops.h.inc>

}  // namespace omtalk
}  // namespace mlir

///
/// Omtalk Dialect
///

namespace omtalk {

using namespace mlir::omtalk;

class Dialect : public mlir::Dialect {
 public:
  explicit Dialect(mlir::MLIRContext *ctx);

  static llvm::StringRef getDialectNamespace() { return "omtalk"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
};

///
/// Omtalk Types
///

namespace OmtalkTypes {
enum Types {
  Box = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  BoxInt,
  BoxRef,
};
}  // namespace OmtalkTypes

class BoxType : public mlir::Type::TypeBase<BoxType, mlir::Type> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == OmtalkTypes::Box; }

  static BoxType get(mlir::MLIRContext *context) {
    return Base::get(context, OmtalkTypes::Box);
  }
};

class BoxIntType : public mlir::Type::TypeBase<BoxIntType, mlir::Type> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == OmtalkTypes::BoxInt; }

  static BoxIntType get(mlir::MLIRContext *context) {
    return Base::get(context, OmtalkTypes::BoxInt);
  }
};

class BoxRefType : public mlir::Type::TypeBase<BoxRefType, mlir::Type> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == OmtalkTypes::BoxRef; }

  static BoxRefType get(mlir::MLIRContext *context) {
    return Base::get(context, OmtalkTypes::BoxRef);
  }
};

}  // namespace omtalk

#endif  // OMTALK_DIALECT_HPP_
