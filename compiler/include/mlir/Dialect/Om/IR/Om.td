//===- OmOps.td - Om dialect Ops         -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OM_OPS
#define OM_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Om dialect
//===----------------------------------------------------------------------===//

def Om_Dialect : Dialect {
  let name = "om";

  let summary = "Object Model Dialect";

  let description = [{
    This is a dialect for representing garbage collected datastructures.
  }];

  let cppNamespace = "om";
}

//===----------------------------------------------------------------------===//
// Base Om operation definition
//===----------------------------------------------------------------------===//

class Om_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Om_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Om Definitions
//===----------------------------------------------------------------------===//

def Om_EndOp : Om_Op<"end", [Terminator]> {
  let summary = "Terminator for datatypes in OM.";
}

def Om_RecordOp : Om_Op<"record", [
    IsolatedFromAbove, Symbol, SymbolTable,
    NoRegionArguments,
    SingleBlockImplicitTerminator<"EndOp">
  ]> {

  let summary = "Define a record type.";

  let description = [{
    A struct is a record type with a fixed layout.
    Fields may be rearranged 
  }];
  let arguments = (ins StrAttr:$sym_name, StrAttr:$super);
  let regions = (region SizedRegion<1>:$body);
}

def Omtalk_FieldOp : Om_Op<"field", [
     HasParent<"RecordOp">, Symbol
    ]> {
  let summary = "Define an instance variable, aka field";
  let arguments = (ins StrAttr:$sym_name);
  let results = (outs);
}

#endif // OM_OPS
