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

//===----------------------------------------------------------------------===//
// Omtalk Operations
//===----------------------------------------------------------------------===//

def ConstantRefOp : Omtalk_Op<"constant_ref", [NoSideEffect]> {
  let summary = "constant reference";
  let description = [{
    A constant boxed object reference.
  }];
  let arguments = (ins I64Attr:$value);
  let results = (outs I64);
}

def ConstantIntOp : Omtalk_Op<"constant_int", [NoSideEffect]> {
  let summary = "constant integer";
  let description = [{
    A constant boxed integer.
  }];
  let arguments = (ins I64Attr:$value);
  let results = (outs I64);
}

// def SendOp : Omtalk_Op<"send",
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

#endif // OMTALK_OPS
