#ifndef OMTALK_IR_OMTALKTYPES_H
#define OMTALK_IR_OMTALKTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

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

#endif // OMTALK_IR_OMTALKTYPES_H