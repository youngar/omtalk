#include <cstdlib>
#include <llvm/Support/CommandLine.h>
#include <mlir/Dialect/Omtalk/IR/OmtalkDialect.h>
#include <mlir/Dialect/Omtalk/IR/OmtalkOps.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR.h>
#include <mlir/Transforms/Passes.h>
#include <omtalk/IRGen/IRGen.h>
#include <omtalk/Parser/AST.h>
#include <omtalk/Parser/Debug.h>
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

static cl::opt<bool> enableKlassResolution("res",
                                           cl::desc("Enable class loading"),
                                           cl::init(true));

cl::opt<std::string> klassPath("path", cl::desc("Append to the class path"), cl::value_desc("directory"));

int main(int argc, char **argv) {

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "Omtalk Parser\n");

  auto ast = omtalk::parser::parseFile(inputFilename);

  if (emitAction == Action::DumpAST) {
    omtalk::parser::print(std::cout, *ast);
    return EXIT_SUCCESS;
  }

  if (emitAction == Action::DumpMLIR) {
    mlir::registerDialect<mlir::omtalk::OmtalkDialect>();
    mlir::MLIRContext context;
    mlir::OwningModuleRef module = omtalk::irgen::irGen(context, *ast);
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
