#ifndef OMTALK_IR_OMTALKOPS_H
#define OMTALK_IR_OMTALKOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// ODS Op Definitions.
#define GET_OP_CLASSES
#include "omtalk/IR/OmtalkOps.h.inc"

#endif // OMTALK_IR_OMTALKOPS_H
