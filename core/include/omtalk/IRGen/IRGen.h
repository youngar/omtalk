#ifndef OMTALK_IRGEN_IRGEN_H_
#define OMTALK_IRGEN_IRGEN_H_

#include <omtalk/Parser/AST.h>

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace omtalk {
namespace irgen {

mlir::OwningModuleRef irGen(mlir::MLIRContext &context,
                            parser::ClassDecl &classDecl);

}
} // namespace omtalk

#endif
