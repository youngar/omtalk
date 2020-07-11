#ifndef OMTALK_IRGEN_IRGEN_H_
#define OMTALK_IRGEN_IRGEN_H_

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <omtalk/Parser/AST.h>

namespace omtalk {
namespace irgen {

mlir::OwningModuleRef irGen(mlir::MLIRContext &context, parser::Module &module);

}
} // namespace omtalk

#endif
