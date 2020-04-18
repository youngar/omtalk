#include <algorithm>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/Omtalk/Passes.hpp"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

namespace {
class DeadFunctionEliminationPass
    : public mlir::ModulePass<DeadFunctionEliminationPass> {
 public:
  void runOnModule() override {
    mlir::ModuleOp module = getModule();
    mlir::SymbolTable moduleSymTable(module);

    // Eliminate non-main functions.
    auto mainFn = moduleSymTable.lookup<mlir::FuncOp>("run");
    for (mlir::FuncOp func :
         llvm::make_early_inc_range(module.getOps<mlir::FuncOp>())) {
      if (func != mainFn) func.erase();
    }
  }
};

}  // end anonymous namespace

/// Create a pass that eliminates inlined functions in toy.
std::unique_ptr<mlir::Pass> omtalk::createDeadFunctionEliminationPass() {
  return std::make_unique<DeadFunctionEliminationPass>();
}
