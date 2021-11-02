#ifndef OM_IR_OMOPS_H
#define OM_IR_OMOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// ODS Op Definitions.
#define GET_OP_CLASSES
#include "om/IR/OmOps.h.inc"

#endif // OM_IR_OMOPS_H
