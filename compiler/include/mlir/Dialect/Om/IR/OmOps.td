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
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Om dialect definition.
//===----------------------------------------------------------------------===//

def Om_Dialect : Dialect {
  let name = "Om";
  let summary = "Object Model Dialect";
  let description = [{
    This is a dialect for representing garbage collected datastructures.
  }];
  let cppNamespace = "omtalk::om";

  let extraClassDeclaration = [{
  }];
}

//===----------------------------------------------------------------------===//
// Base Om operation definition
//===----------------------------------------------------------------------===//

class Om_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Om_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Om Definitions
//===----------------------------------------------------------------------===//

def Om_EndOp : Om_Op<"end", [Terminator, HasParent<Om_RecordOp>]> {
  let summary = "Terminator for datatypes in OM."
}

def Omtalk_FieldOp : Omtalk_Op<"field", [
     HasParent<"KlassOp">, Symbol
    ]> {
  let summary = "Define an instance variable, aka field";
  let arguments = (ins StrAttr:$sym_name);
}

def Omtalk_KlassEndOp : Omtalk_Op<"klass_end", [
   HasParent<"KlassOp">, Terminator
  ]> {
  let summary = "Psuedo instruction terminating the body of a class";
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

  let parser = [{
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, "sym_name", result.attributes)) {
      return failure();
    }

    StringAttr superAttr;
    if (succeeded(parser.parseOptionalColon())) {
      if (parser.parseSymbolName(superAttr, "super", result.attributes)) {
        return failure();
      }
    }

    auto *body = result.addRegion();
    if (parser.parseRegion(*body, llvm::None, llvm::None)) {
      return failure();
    }

    ensureTerminator(*body, parser.getBuilder(), result.location);
    return success();
  }];

  let printer = [{
    p << KlassOp::getOperationName();
    p << " ";
    p.printSymbolName(sym_name());
    p << " : ";
    p.printSymbolName(super());
    p.printRegion(body(), false, false);
  }];

  let extraClassDeclaration = [{
    void push_back(Operation *op) {
      body().front().push_back(op);
    }
  }];
}

#endif // OM_OPS
