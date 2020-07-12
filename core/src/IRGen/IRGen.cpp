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
    // std::cout << loc;
    return builder.getFileLineColLoc(builder.getIdentifier(loc.filename),
                                     loc.start.line, loc.start.line);
  }

  mlir::omtalk::KlassOp irGen(const KlassDecl &klassDecl) {
    auto klassOp = builder.create<mlir::omtalk::KlassOp>(loc(klassDecl.location),
                                                         klassDecl.name);
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
