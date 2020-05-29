#ifndef OMTALK_PASSES_HPP_
#define OMTALK_PASSES_HPP_

#include <memory>

namespace mlir {
class Pass;
}

namespace omtalk {
std::unique_ptr<mlir::Pass> createDeadFunctionEliminationPass();
std::unique_ptr<mlir::Pass> createLowerPass();
std::unique_ptr<mlir::Pass> createToLlvmLoweringPass();
}

#endif  // OMTALK_PASSES_HPP_