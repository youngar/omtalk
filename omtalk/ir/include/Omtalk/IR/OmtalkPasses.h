#ifndef OMTALK_IR_OMTALKPASSES_H
#define OMTALK_IR_OMTALKPASSES_H

#include <memory>

namespace mlir {
class Pass;
}

namespace omtalk {
std::unique_ptr<mlir::Pass> createDeadFunctionEliminationPass();
std::unique_ptr<mlir::Pass> createLowerPass();
std::unique_ptr<mlir::Pass> createToLlvmLoweringPass();
} // namespace omtalk

#endif // OMTALK_IR_OMTALKPASSES_H