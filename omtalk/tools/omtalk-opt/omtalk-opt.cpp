#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "om/IR/OmDialect.h"
#include "omtalk/IR/OmtalkDialect.h"
#include "omtalk/IR/OmtalkPasses.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all dialects.
  registry.insert<om::OmDialect>();
  registry.insert<omtalk::OmtalkDialect>();

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "Omtalk modular optimizer driver", registry,
                        /*preloadDialectsInContext=*/false));
}
