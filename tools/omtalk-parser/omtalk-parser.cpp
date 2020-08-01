#include <cstdlib>
#include <llvm/Support/CommandLine.h>

#include <mlir/Support/LLVM.h>

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
#include <omtalk/KlassLoader.h>
#include <omtalk/Parser/AST.h>
#include <omtalk/Parser/Debug.h>
#include <omtalk/Parser/Location.h>
#include <omtalk/Parser/Parser.h>

using namespace llvm;
using namespace mlir;

using namespace ::omtalk;
namespace cl = llvm::cl;

static cl::OptionCategory omtalkCategory("Omtalk Options");

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"),
                  cl::value_desc("filename"), cl::cat(omtalkCategory));

namespace {
enum Action {
  None,
  EmitAST,
  EmitMLIR,
};
}

static cl::opt<enum Action>
    emitAction("emit", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(EmitAST, "ast", "output the AST dump")),
               cl::values(clEnumValN(EmitMLIR, "mlir", "output the MLIR dump")),
               cl::cat(omtalkCategory));

static cl::list<std::string> classPath("cp", cl::desc("Class search path"),
                                       cl::cat(omtalkCategory));

std::vector<std::string> getClassPath() {
  std::vector<std::string> path;
  for (const auto &dir : classPath) {
    path.push_back(dir);
  }
  return path;
}

int main(int argc, char **argv) {

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "Omtalk Parser\n");

  ::omtalk::KlassLoader loader(getClassPath());

  if (!loader.loadFileOrKlass(inputFilename)) {
    llvm::outs() << "error: failed to load file or klass\n";
    return EXIT_FAILURE;
  }

  if (emitAction == Action::EmitAST) {
    for (const auto &module : loader.getModules()) {
      ::omtalk::parser::print(std::cout, *module);
    }
    return EXIT_SUCCESS;
  }

  if (emitAction == Action::EmitMLIR) {
    mlir::registerDialect<mlir::omtalk::OmtalkDialect>();
    mlir::MLIRContext context;
    mlir::OwningModuleRef module = ::omtalk::irgen::irGen(context, loader.getModules());
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
