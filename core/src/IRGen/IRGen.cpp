#include <mlir/Dialect/Omtalk/IR/OmtalkDialect.h>
#include <mlir/Dialect/Omtalk/IR/OmtalkOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Verifier.h>
#include <omtalk/IRGen/IRGen.h>
#include <omtalk/Parser/Location.h>

#include <iostream>

using namespace omtalk;
using namespace omtalk::irgen;
using namespace omtalk::parser;

namespace {

// void GPUModuleOp::build(OpBuilder &builder, OperationState &result,
//                         StringRef name) {
//   ensureTerminator(*result.addRegion(), builder, result.location);
//   result.attributes.push_back(builder.getNamedAttr(
//       ::mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
// }

class IRGen {
public:
  IRGen(mlir::MLIRContext &context) : builder(&context) {}

  mlir::Location loc(Location loc) {
    return builder.getFileLineColLoc(builder.getIdentifier(loc.filename),
                                     loc.start.line, loc.start.line);
  }

  mlir::omtalk::KlassOp irGen(const KlassDecl &klassDecl) {
    
    std::string super = "Object";
    if (klassDecl.super) {
      super = klassDecl.super->value;
    }

    auto klassOp = builder.create<mlir::omtalk::KlassOp>(
        loc(klassDecl.location), klassDecl.name.value, super);

    mlir::omtalk::KlassOp::ensureTerminator(klassOp.body(), builder, loc(klassDecl.location));
    auto endOp =
        builder.create<mlir::omtalk::KlassEndOp>(loc(klassDecl.location));

    // klassOp.body().getOperations().push_back(endOp);

    return klassOp;
  }

  mlir::ModuleOp irGen(Module &module) {

    moduleOp = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (const auto &klass : module.klassDecls) {
      moduleOp.push_back(irGen(*klass));
    }

    if (mlir::failed(mlir::verify(moduleOp))) {
      moduleOp.emitError("module verification error");
      return nullptr;
    }

    return moduleOp;
  }

private:
  mlir::ModuleOp moduleOp;
  mlir::OpBuilder builder;
};

} // namespace

mlir::OwningModuleRef omtalk::irgen::irGen(mlir::MLIRContext &context,
                                           parser::Module &module) {

  IRGen irGen(context);
  return irGen.irGen(module);
}
