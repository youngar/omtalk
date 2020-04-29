
#include <numeric>

#include "mlir/Dialect/Omtalk/IR/OmtalkDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

namespace omtalk {

namespace {
#include "Transforms/Canonicalize.inc"
}

}  // namespace omtalk
}  // namespace mlir