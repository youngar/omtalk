#include <mlir/IR/Verifier.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <omtalk/IRGen/IRGen.h>

using namespace omtalk;
using namespace omtalk::irgen;
using namespace omtalk::parser;

namespace {

class IRGenImpl {
public:
  IRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp irGen(ClassDecl &classDecl) {

    // create an empty module
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    if (mlir::failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
};
} // namespace

mlir::OwningModuleRef omtalk::irgen::irGen(mlir::MLIRContext &context,
                                           parser::ClassDecl &classDecl) {

  IRGenImpl irGen(context);
  return irGen.irGen(classDecl);
}
