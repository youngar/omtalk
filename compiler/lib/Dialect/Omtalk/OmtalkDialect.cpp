#include "mlir/Dialect/Omtalk/IR/OmtalkDialect.h"
#include "mlir/Dialect/Omtalk/IR/OmtalkOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::omtalk;

namespace mlir {
namespace omtalk {

//===----------------------------------------------------------------------===//
// Omtalk dialect.
//===----------------------------------------------------------------------===//

OmtalkDialect::OmtalkDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Omtalk/IR/OmtalkOps.cpp.inc"
      >();
  // addTypes<BoxType, BoxIntType, BoxRefType>();
}

//===----------------------------------------------------------------------===//
// Omtalk types.
//===----------------------------------------------------------------------===//

// mlir::Type OmtalkDialect::parseType(mlir::DialectAsmParser &parser) const {
//   if (parser.parseKeyword("box") || parser.parseLess())
//     return mlir::Type();

//   if (parser.parseKeyword("int") || parser.parseGreater())
//     return BoxIntType::get(getContext());

//   if (parser.parseKeyword("ref") || parser.parseGreater())
//     return BoxRefType::get(getContext());

//   if (parser.parseOptionalQuestion() || parser.parseGreater())
//     return BoxType::get(getContext());

//   return mlir::Type();
// }

// void OmtalkDialect::printType(mlir::Type type,
//                               mlir::DialectAsmPrinter &printer) const {
//   // BoxType boxType = type.cast<BoxType>();
//   switch (type.getKind()) {
//   default:
//     llvm_unreachable("Unhandled Linalg type");
//     break;
//   case OmtalkTypes::Box:
//     printer << "box<?>";
//     break;
//   case OmtalkTypes::BoxInt:
//     printer << "box<int>";
//     break;
//   case OmtalkTypes::BoxRef:
//     printer << "box<ref>";
//     break;
//   }
// }

} // namespace omtalk
} // namespace mlir