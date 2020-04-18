
#include <numeric>

#include "mlir/Dialect/Omtalk/OmtalkDialect.hpp"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

namespace omtalk {

namespace {
#include "Transforms/Canonicalize.inc"
}

void SendOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<ReplaceSendInt>(context);
}

}  // namespace omtalk
}  // namespace mlir