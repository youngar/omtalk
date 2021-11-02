#include <mlir/IR/DialectImplementation.h>
#include <omtalk/IR/OmtalkDialect.h>
#include <omtalk/IR/OmtalkTypes.h>

using namespace omtalk;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Omtalk types.
//===----------------------------------------------------------------------===//

mlir::Type OmtalkDialect::parseType(DialectAsmParser &parser) const {
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

void OmtalkDialect::printType(Type type, DialectAsmPrinter &printer) const {
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

void OmtalkDialect::registerTypes() {
  addTypes<BoxUnkType, BoxIntType, BoxRefType>();
}
