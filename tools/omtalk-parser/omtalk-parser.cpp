#include <cstdlib>
#include <llvm/Support/CommandLine.h>
#include <mlir/Analysis/Verifier.h>
#include <mlir/Dialect/Omtalk/IR/OmtalkDialect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR.h>
#include <mlir/Transforms/Passes.h>
#include <omtalk/IRGen/IRGen.h>
#include <omtalk/Parser/AST.h>
#include <omtalk/Parser/Location.h>
#include <omtalk/Parser/Parser.h>

using namespace omtalk;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum Action {
  None,
  DumpAST,
  DumpMLIR,
};
}

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

int main(int argc, char **argv) {
  mlir::registerAllDialects();
  // mlir::registerDialect<mlir::omtalk::OmtalkDialect>();

  cl::ParseCommandLineOptions(argc, argv, "Omtalk Parser\n");

  auto classDecl = omtalk::parser::parseClassFile(inputFilename);

  mlir::MLIRContext context;

  if (emitAction == Action::DumpAST) {
    omtalk::parser::dump(*classDecl);
    return EXIT_SUCCESS;
  }

  mlir::OwningModuleRef module = omtalk::irgen::irGen(context, *classDecl);

  if (emitAction == Action::DumpMLIR) {
    module->dump();
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
