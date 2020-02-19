
#include <llvm/Support/CommandLine.h>

#include <iostream>
#include <mlir/Dialect/Omtalk/OmtalkDialect.hpp>
#include "mlir/Pass/PassManager.h"

namespace cl = llvm::cl;

static cl::opt<bool> verbose("v", cl::desc("Enable verbose output"));

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int main(int argc, char **argv) {
  mlir::registerPassManagerCLOptions();
  // applyPassManagerCLOptions(pm);
  
  cl::ParseCommandLineOptions(argc, argv, "Omtalk");

  std::cout << "input file: " << inputFilename << std::endl;

  return 0;
}
