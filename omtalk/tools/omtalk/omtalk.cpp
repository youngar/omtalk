
#include <cstdlib>
#include <llvm/Support/CommandLine.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <om/Dialect/Omtalk/IR/OmtalkDialect.h>
#include <om/Dialect/Omtalk/IR/OmtalkOps.h>
#include <omtalk/IRGen/IRGen.h>
#include <omtalk/Omtalk.h>
#include <omtalk/Parser/AST.h>
#include <omtalk/Parser/Debug.h>
#include <omtalk/Parser/Location.h>
#include <omtalk/Parser/Parser.h>
#include <omtalk/Runtime.h>

using namespace omtalk;
namespace cl = llvm::cl;

static cl::OptionCategory omtalkCategory("Omtalk Options");

static cl::list<std::string> classPath("cp", cl::desc("Class search path"),
                                       cl::cat(omtalkCategory));

namespace {
enum Action { None, EmitAST, EmitMLIR, EmitBin, RunJIT };
}

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(EmitAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(EmitMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(EmitBin, "bin", "output a binary executable")),
    cl::values(
        clEnumValN(RunJIT, "jit",
                   "JIT the code and run it by invoking the main function")),
    cl::cat(omtalkCategory));

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"),
                  cl::value_desc("filename"), cl::cat(omtalkCategory));

int main(int argc, char **argv) {

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "Omtalk\n");

  omtalk::VirtualMachineConfig config;

  omtalk::Process process;
  omtalk::Thread thread(process);
  omtalk::VirtualMachine vm(thread, config);

  omtalk::bootstrap(vm);

  auto ast = omtalk::parser::parseFile(inputFilename);

  if (emitAction == Action::EmitAST) {
    omtalk::parser::print(std::cout, *ast);
    return EXIT_SUCCESS;
  }

  if (emitAction == Action::EmitMLIR) {
    mlir::MLIRContext context;
    context.loadDialect<om::omtalk::OmtalkDialect>();
    mlir::OwningModuleRef module = omtalk::irgen::irGen(context, *ast);
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
