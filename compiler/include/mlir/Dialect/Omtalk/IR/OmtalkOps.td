//===- OmtalkOps.td - Omtalk dialect ops -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OMTALK_OPS
#define OMTALK_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Omtalk dialect definition.
//===----------------------------------------------------------------------===//

def Omtalk_Dialect : Dialect {
  let name = "omtalk";
  let summary = "Omtalk Smalltalk Dialect";
  let description = [{
      This is a dialect for representing small talk objects. It is tightly
      coupled to the omtalk virtual machine.
  }];
  let cppNamespace = "omtalk";
}

//===----------------------------------------------------------------------===//
// Base omtalk operation definition.
//===----------------------------------------------------------------------===//

class Omtalk_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Omtalk_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Omtalk Types
//===----------------------------------------------------------------------===//

// def Omtalk_BoxUnkType :
//     Type<CPred<"$_self.isa<::omtalk::BoxType>()">, "Omtalk Boxed Type">;

// def Omtalk_BoxIntType :
//     Type<CPred<"$_self.isa<::omtalk::BoxIntType>()">, "Omtalk Boxed Primitive Integer Type">;

// def Omtalk_BoxRefType :
//     Type<CPred<"$_self.isa<::omtalk::BoxRefType>()">, "Omtalk Boxed Reference Type">;

// def Omtalk_AnyBoxType : AnyTypeOf<[Omtalk_BoxUnkType, Omtalk_BoxIntType, Omtalk_BoxRefType]>;

// def Omtalk_Type : AnyTypeOf<[Omtalk_AnyBoxType]>;

def Omtalk_Type : AnyTypeOf<[I64]>;

//===----------------------------------------------------------------------===//
// Omtalk Operations
//===----------------------------------------------------------------------===//

def Omtalk_ConstantRefOp : Omtalk_Op<"constant_ref", [NoSideEffect]> {
  let summary = "constant reference";
  let description = [{
    A constant boxed object reference.
  }];
  let arguments = (ins I64Attr:$value);
  let results = (outs I64);
}

def Omtalk_ConstantIntOp : Omtalk_Op<"constant_int", [NoSideEffect]> {
  let summary = "constant integer";
  let description = [{
    A constant boxed integer.
  }];
  let arguments = (ins I64Attr:$value);
  let results = (outs I64);
}

// def Omtalk_SendOp : Omtalk_Op<"send",
//     [DeclareOpInterfaceMethods<CallOpInterface>]> {
//   let summary = "Send message to receiver (generic)";
//   let description = [{
//     A generic send operation, where the receiver parameter may be a boxed integer or boxed reference.
//     This operation is lowered to the more specific send_int and send_ref operations. 
//   }];

//   let arguments = (ins Omtalk_Type:$recv, FlatSymbolRefAttr:$message, Variadic<Omtalk_Type>:$inputs);
//   let results = (outs Omtalk_Type);
//   let hasCanonicalizer = 1;
// }

//===----------------------------------------------------------------------===//
// Omtalk Definitions
//===----------------------------------------------------------------------===//

def Omtalk_KlassOp : Omtalk_Op<"klass", [
    IsolatedFromAbove, Symbol, SymbolTable,
    SingleBlockImplicitTerminator<"KlassEndOp">
  ]> {

  let summary = "Define a klass";
  let arguments = (ins StrAttr:$sym_name, StrAttr:$super);
  let regions = (region SizedRegion<1>:$body);

//  let builders = [
//    OpBuilder<
//      "OpBuilder &builder, OperationState &result, "
//      "StringRef name">
//  ];

  let parser = [{
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, "sym_name", result.attributes)) {
      return failure();
    }

    // TODO: does this work at all???????
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

def Omtalk_KlassEndOp : Omtalk_Op<"klass_end", [
   Terminator, HasParent<"KlassOp">
  ]> {
  let summary = "Psuedo instruction terminating the body of a class";
}

def Omtalk_VarOp : Omtalk_Op<"var", [Symbol]> {
  let summary = "Define an instance variable";
  let arguments = (ins StrAttr:$sym_name);
}

def Omtalk_MethodOp : Omtalk_Op<"method", [
    IsolatedFromAbove, Symbol
  ]> {
  let summary = "Define a method on an instance";
  let arguments = (ins StrAttr:$sym_name);
  // let regions = (region SizedRegion<1>:$body);
}

//===----------------------------------------------------------------------===//
// Omtalk Operations
//===----------------------------------------------------------------------===//

def Omtalk_SelfOp : Omtalk_Op<"self", [NoSideEffect]> {
  let summary = "Get a pointer to the current receiver";
  let results = (outs Omtalk_Type);
}

def Omtalk_LoadClassOp : Omtalk_Op<"load_class", [NoSideEffect]> {
  let summary = "Load a reference to a named class";
  let arguments = (ins SymbolRefAttr:$name);
  let results = (outs Omtalk_Type);
}

def Omtalk_LoadVarOp : Omtalk_Op<"load_var", [NoSideEffect]> {
  let summary = "Load a reference to a named variable";
  let arguments = (ins SymbolRefAttr:$name);
  let results = (outs Omtalk_Type);
}

def Omtalk_StoreVarOp : Omtalk_Op<"store_var", [NoSideEffect]> {
  let summary = "Load a reference to a named variable";
  let arguments = (ins SymbolRefAttr:$name, Omtalk_Type:$value);
}

#endif // OMTALK_OPS
