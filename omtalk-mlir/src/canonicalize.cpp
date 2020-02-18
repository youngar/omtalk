
#include <numeric>
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "omtalk/dialect.hpp"

namespace mlir {

namespace omtalk {

namespace {
#include "canonicalize.inc"
}

void SendOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<ReplaceSendInt>(context);
}

}  // namespace omtalk
}  // namespace mlir