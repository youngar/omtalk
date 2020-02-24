#include "mlir/Dialect/Omtalk/OmtalkDialect.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace omtalk {

//===----------------------------------------------------------------------===//
// SendOp
//===----------------------------------------------------------------------===//

mlir::CallInterfaceCallable SendOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("message");
}

mlir::Operation::operand_range SendOp::getArgOperands() { return inputs(); }

//===----------------------------------------------------------------------===//
// SendIntOp
//===----------------------------------------------------------------------===//

mlir::CallInterfaceCallable SendIntOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("message");
}

mlir::Operation::operand_range SendIntOp::getArgOperands() { return inputs(); }

//===----------------------------------------------------------------------===//
// SendIntAddOp
//===----------------------------------------------------------------------===//

mlir::CallInterfaceCallable SendIntAddOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("message");
}

mlir::Operation::operand_range SendIntAddOp::getArgOperands() {
  return inputs();
}

OpFoldResult SendIntAddOp::fold(ArrayRef<Attribute> operands) {
  /// addi(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a + b; });
}

//===----------------------------------------------------------------------===//
// SendRef
//===----------------------------------------------------------------------===//

mlir::CallInterfaceCallable SendRefOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("message");
}

mlir::Operation::operand_range SendRefOp::getArgOperands() { return inputs(); }

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "omtalk/ops.cpp.inc"

}  // namespace omtalk
}  // namespace mlir
