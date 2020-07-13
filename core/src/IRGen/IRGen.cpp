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

    {
      mlir::OpBuilder::InsertionGuard guard(builder);

      auto region = &klassOp.body();
      auto block = builder.createBlock(region);

      builder.setInsertionPointToStart(block);

      for (const auto &field : klassDecl.fields.elements) {
        builder.create<mlir::omtalk::FieldOp>(loc(field.location), field.value);
      }

      builder.create<mlir::omtalk::KlassEndOp>(loc(klassDecl.location));
    }
  
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
