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

  /// This hook is called when a terminator operation has been inlined. The only
  /// terminator that we have in the Toy dialect is the return
  /// operation(toy.return). We handle the return by replacing the values
  /// previously returned by the call operation with the operands of the
  /// return.
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
  addTypes<BoxType>();
  addInterfaces<InlinerInterface>();
}

///
/// Type Parser and Printer
///

mlir::Type Dialect::parseType(mlir::DialectAsmParser &parser) const {
  //
  // Box Types
  //
  if (parser.parseKeyword("box") || parser.parseLess() ||
      parser.parseOptionalQuestion() || parser.parseGreater())
    return mlir::Type();

  return BoxType::get(getContext());
}

void Dialect::printType(mlir::Type type,
                        mlir::DialectAsmPrinter &printer) const {
  BoxType boxType = type.cast<BoxType>();
  printer << "box<?>";
}

}  // namespace omtalk

namespace mlir {
namespace omtalk {

/// Return the callee of the generic call operation, this is required by the
/// call interface.
mlir::CallInterfaceCallable StaticSendOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("message");
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
mlir::Operation::operand_range StaticSendOp::getArgOperands() {
  return inputs();
}

#define GET_OP_CLASSES
#include "omtalk/ops.cpp.inc"
}  // namespace omtalk
}  // namespace mlir
