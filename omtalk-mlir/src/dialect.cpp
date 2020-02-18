#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Transforms/InliningUtils.h>

#include <omtalk/dialect.hpp>

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"

namespace omtalk {

struct InlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(mlir::Operation *, mlir::Region *,
                       mlir::BlockAndValueMapping &) const final {
    // All blocks are inlinable
    return true;
  }

  void handleTerminator(mlir::Operation *op,
                        llvm::ArrayRef<mlir::Value> valuesToRepl) const final {

    // // Only "toy.return" needs to be handled here.
    auto returnOp = llvm::cast<omtalk::ReturnOp>(op);

    // Replace the values directly with the return operands.
    // assert(returnOp.getNumOperands() == valuesToRepl.size());
    //     for (const auto &it : llvm::enumerate(returnOp.getOperands())
    //       valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    //   }

    valuesToRepl[0].replaceAllUsesWith(returnOp.getOperand());
  }
};

Dialect::Dialect(mlir::MLIRContext *ctx) : mlir::Dialect("omtalk", ctx) {
#define GET_OP_LIST
  addOperations<
#include "omtalk/ops.cpp.inc"
      >();
  addTypes<BoxType, BoxIntType, BoxRefType>();
  addInterfaces<InlinerInterface>();
}

///
/// Type Parser and Printer
///

mlir::Type Dialect::parseType(mlir::DialectAsmParser &parser) const {

  if (parser.parseKeyword("box") || parser.parseLess())
    return mlir::Type();

  if(parser.parseKeyword("int") || parser.parseGreater())
    return BoxIntType::get(getContext());

  if(parser.parseKeyword("ref") || parser.parseGreater())
    return BoxRefType::get(getContext());

  if(parser.parseOptionalQuestion() || parser.parseGreater())
    return BoxType::get(getContext());
  
  return mlir::Type();
}

void Dialect::printType(mlir::Type type,
                        mlir::DialectAsmPrinter &printer) const {

  // BoxType boxType = type.cast<BoxType>();
  switch (type.getKind()) {
  default:
    llvm_unreachable("Unhandled Linalg type");
    break;
  case OmtalkTypes::Box:
    printer << "box<?>";
    break;
  case OmtalkTypes::BoxInt:
    printer << "box<int>";
    break;
  case OmtalkTypes::BoxRef:
    printer << "box<ref>";
    break;
  }
}

}  // namespace omtalk

namespace mlir {
namespace omtalk {

mlir::CallInterfaceCallable SendOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("message");
}

mlir::Operation::operand_range SendOp::getArgOperands() {
  return inputs();
}

mlir::CallInterfaceCallable SendIntOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("message");
}

mlir::Operation::operand_range SendIntOp::getArgOperands() {
  return inputs();
}

mlir::CallInterfaceCallable SendRefOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("message");
}

mlir::Operation::operand_range SendRefOp::getArgOperands() {
  return inputs();
}


#define GET_OP_CLASSES
#include "omtalk/ops.cpp.inc"
}  // namespace omtalk
}  // namespace mlir
