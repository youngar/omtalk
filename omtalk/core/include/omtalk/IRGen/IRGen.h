#ifndef OMTALK_IRGEN_IRGEN_H_
#define OMTALK_IRGEN_IRGEN_H_

#include <mlir/IR/MLIRContext.h>
#include <omtalk/Parser/AST.h>

namespace omtalk::irgen {

mlir::OwningModuleRef irGen(mlir::MLIRContext &context, parser::Module &module);

mlir::OwningModuleRef irGen(mlir::MLIRContext &context,
                            const std::vector<parser::ModulePtr> &modules);

} // namespace omtalk::irgen

#endif
