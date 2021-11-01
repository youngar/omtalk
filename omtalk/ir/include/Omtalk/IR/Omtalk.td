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
include "mlir/Interfaces/ControlFlowInterfaces.td"
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

  let extraClassDeclaration = [{
    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type type,
                   mlir::DialectAsmPrinter &printer) const override;
  }];

}

//===----------------------------------------------------------------------===//
// Base omtalk operation definition.
//===----------------------------------------------------------------------===//

class Omtalk_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Omtalk_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Omtalk Types
//===----------------------------------------------------------------------===//

def Omtalk_BoxUnkType :
    Type<CPred<"$_self.isa<::omtalk::BoxUnkType>()">, "!omtalk.box<?>">;

def Omtalk_BoxIntType :
    Type<CPred<"$_self.isa<::omtalk::BoxIntType>()">, "!omtalk.box<int>">;

def Omtalk_BoxRefType :
    Type<CPred<"$_self.isa<::omtalk::BoxRefType>()">, "!omtalk.box<ref>">;

def Omtalk_AnyBoxType : AnyTypeOf<[Omtalk_BoxUnkType, Omtalk_BoxIntType, Omtalk_BoxRefType]>;

def Omtalk_Type : AnyTypeOf<[Omtalk_AnyBoxType]>;

//===----------------------------------------------------------------------===//
// Omtalk Definitions
//===----------------------------------------------------------------------===//

def Omtalk_KlassOp : Omtalk_Op<"klass", [
    IsolatedFromAbove, Symbol, SymbolTable,
    NoRegionArguments,
    SingleBlockImplicitTerminator<"KlassEndOp">
  ]> {

  let summary = "Define a klass";
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
    return mlir::success();
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
    void push_back(mlir::Operation *op) {
      body().front().push_back(op);
    }
  }];
}

def Omtalk_KlassEndOp : Omtalk_Op<"klass_end", [
   HasParent<"KlassOp">, Terminator
  ]> {
  let summary = "Psuedo instruction terminating the body of a class";
}

def Omtalk_FieldOp : Omtalk_Op<"field", [
     HasParent<"KlassOp">, Symbol
    ]> {
  let summary = "Define an instance variable, aka field";
  let arguments = (ins StrAttr:$sym_name);
}

def Omtalk_MethodOp : Omtalk_Op<"method", [
    HasParent<"KlassOp">, AutomaticAllocationScope, FunctionLike,
    IsolatedFromAbove, Symbol
  ]> {

  let summary = "Define a method on an instance";

  let arguments = (ins
    TypeAttr:$type,
    StrAttr:$sym_name
  );

  let results = (outs);

  let regions = (region AnyRegion:$body);

  //let skipDefaultBuilders = 1;

  //let builders = [
  //  OpBuilder<(ins "StringRef":$name, "unsigned":$argCount)>
  //];

  let extraClassDeclaration = [{
    unsigned getNumFuncArguments() {
      return getType().getNumInputs();
    }

    unsigned getNumFuncResults() {
      return getType().getNumResults();
    }

    mlir::LogicalResult verifyType() const { return mlir::success(); }
  }];
}

//===----------------------------------------------------------------------===//
// Omtalk Literals and Constants
//===----------------------------------------------------------------------===//

def Omtalk_ConstantRefOp : Omtalk_Op<"constant_ref", [
    ConstantLike, NoSideEffect
  ]> {
  let summary = "constant reference";
  let description = [{
    A constant boxed object reference.
  }];
  let arguments = (ins I64Attr:$value);
  let results = (outs Omtalk_BoxRefType);
}

def Omtalk_ConstantIntOp : Omtalk_Op<"constant_int", [
    ConstantLike, NoSideEffect
  ]> {
  let summary = "constant integer";
  let description = [{
    A constant boxed integer.
  }];
  let arguments = (ins I64Attr:$value);
  let results = (outs Omtalk_BoxIntType);
}

def Omtalk_ConstantFloatOp : Omtalk_Op<"constant_float", [
    ConstantLike, NoSideEffect
  ]> {
  let summary = "constant float";
  let arguments = (ins F64Attr:$value);
  let results = (outs Omtalk_BoxRefType);
}

def Omtalk_ConstantStringOp : Omtalk_Op<"constant_string", [
    ConstantLike, NoSideEffect
  ]> {
  let summary = "constant string";
  let arguments = (ins StrAttr:$value);
  let results = (outs Omtalk_BoxRefType);
}

def Omtalk_ConstantSymbolOp : Omtalk_Op<"constant_symbol", [
    ConstantLike, NoSideEffect
  ]> {
  let summary = "constant symbol";
  let arguments = (ins StrAttr:$value);
  let results = (outs Omtalk_BoxRefType);
}

//===----------------------------------------------------------------------===//
// Omtalk Operations
//===----------------------------------------------------------------------===//

def Omtalk_SendOp : Omtalk_Op<"send", []> {
  let summary = "Send message to receiver (generic)";
  let description = [{
    A generic send operation, where the receiver parameter may be a boxed integer or boxed reference.
    This operation is lowered to the more specific send_int and send_ref operations. 
  }];

  let arguments = (ins Omtalk_AnyBoxType:$recv, FlatSymbolRefAttr:$message, Variadic<Omtalk_AnyBoxType>:$inputs);
  let results = (outs Omtalk_AnyBoxType);
}

def Omtalk_SuperSendOp : Omtalk_Op<"super_send", []> {
  let summary = "Statically send a method to the super class";
  let arguments = (ins Omtalk_AnyBoxType:$recv, FlatSymbolRefAttr:$message, Variadic<Omtalk_AnyBoxType>:$inputs);
  let results = (outs Omtalk_AnyBoxType);
}

def Omtalk_BlockOp : Omtalk_Op<"block", []> {
  let summary = "An inline block";
  let arguments = (ins TypeAttr:$type);
  let results = (outs Omtalk_AnyBoxType);
  let regions = (region AnyRegion:$body);
}

def Omtalk_GetClassOp : Omtalk_Op<"get_class", [NoSideEffect]> {
  let summary = "Load a reference to a named class";
  let arguments = (ins SymbolRefAttr:$name);
  let results = (outs Omtalk_BoxRefType);
}

def Omtalk_GetFieldOp : Omtalk_Op<"get_field", [NoSideEffect]> {
  let summary = "Load from an instance variable, aka field";
  let arguments = (ins SymbolRefAttr:$name);
  let results = (outs Omtalk_AnyBoxType);
}

def Omtalk_SetFieldOp : Omtalk_Op<"set_field", []> {
  let summary = "Store to an instance variable, aka field";
  let arguments = (ins SymbolRefAttr:$name, Omtalk_Type:$value);
}

def Omtalk_ReturnOp : Omtalk_Op<"return", [Terminator]> {
  let summary = "Return a value from a method or block.";
  let arguments = (ins Omtalk_AnyBoxType:$value);
}

def Omtalk_NonlocalReturnOp : Omtalk_Op<"nonlocal_return", [Terminator]> {
  let summary = "Return from the home frame of a block.";

  let description = [{
    This op is only valid within a block. A block has a parent "home" frame,
    which is the method it was constructed in. When an 
  }];

  let arguments = (ins Omtalk_AnyBoxType:$value);
}

#endif // OMTALK_OPS
