#ifndef OMTALK_OMTALKOPS_TD_
#define OMTALK_OMTALKOPS_TD_

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffects.td"
include "mlir/IR/OpBase.td"

//
// Dialect
//

def Omtalk_Dialect : Dialect {
  let name = "omtalk";
  let cppNamespace = "omtalk";
}

class Omtalk_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Omtalk_Dialect, mnemonic, traits>;

//
// Types
//

def Omtalk_BoxUnkType :
    Type<CPred<"$_self.isa<::omtalk::BoxType>()">, "Omtalk Boxed Type">;

def Omtalk_BoxIntType :
    Type<CPred<"$_self.isa<::omtalk::BoxIntType>()">, "Omtalk Boxed Primitive Integer Type">;

def Omtalk_BoxRefType :
    Type<CPred<"$_self.isa<::omtalk::BoxRefType>()">, "Omtalk Boxed Reference Type">;

def Omtalk_AnyBoxType : AnyTypeOf<[Omtalk_BoxUnkType, Omtalk_BoxIntType, Omtalk_BoxRefType]>;

def Omtalk_Type : AnyTypeOf<[Omtalk_AnyBoxType]>;

//
// Operations
//

def ConstantRefOp : Omtalk_Op<"constant_ref", [NoSideEffect]> {
  let summary = "constant reference";
  let description = [{
    A constant boxed object pointer.
  }];
  let arguments = (ins I64Attr:$value);
  let results = (outs Omtalk_BoxRefType);
}

def ConstantIntOp : Omtalk_Op<"constant_int", [NoSideEffect]> {
  let summary = "constant_int";
  let description = [{
      A constant boxed integer
    }];
  let arguments = (ins I64Attr:$value);
  let results = (outs Omtalk_BoxIntType);
}

def SendOp : Omtalk_Op<"send",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "Send message to receiver (generic)";
  let description = [{
    A generic send operation, where the receiver parameter may be a boxed integer or boxed reference.
    This operation is lowered to the more specific send_int and send_ref operations. 
  }];

  let arguments = (ins Omtalk_Type:$recv, FlatSymbolRefAttr:$message, Variadic<Omtalk_Type>:$inputs);
  let results = (outs Omtalk_Type);
  let hasCanonicalizer = 1;
}

def SendIntOp : Omtalk_Op<"send_int",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "Send message to an integer receiver";
  let description = [{
    Send a message, where the receiver is guaranteed to be a boxed integer.
  }];

  let arguments = (ins Omtalk_BoxIntType:$recv, FlatSymbolRefAttr:$message, Variadic<Omtalk_Type>:$inputs);
  let results = (outs Omtalk_Type);
}

def SendIntAddOp : Omtalk_Op<"send_int_add",
  [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "Send \"+\" to an Integer receiver";

  let arguments = (ins Omtalk_BoxIntType:$recv, Variadic<Omtalk_Type>:$inputs);
  let results = (outs Omtalk_BoxIntType);
  
  let hasFolder = 1;
}

def SendRefOp : Omtalk_Op<"send_ref",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "Send message to an object receiver";
  let description = [{
    Send a message, where the receiver is guaranteed to be a regular object reference.
  }];

  let arguments = (ins Omtalk_BoxRefType:$recv, FlatSymbolRefAttr:$message, Variadic<Omtalk_Type>:$inputs);
  let results = (outs Omtalk_Type);
}

def ReturnOp : Omtalk_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  let summary = "return to the sender";
  let description = [{
    The "return" operation represents a return from the sender to the receiver.
    The receiver always returns a value for a message expression.
  }];

  let arguments = (ins Omtalk_Type:$value);
}

def IAddOp : Omtalk_Op<"iadd", [NoSideEffect]> {
  let summary = "Integer add operator";
  let description = [{
    The "IAdd" operation adds two integer constants together.
  }];

  let arguments = (ins Omtalk_Type:$recv, Omtalk_Type:$rhs);
  let results = (outs Omtalk_Type);

  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs">
  ];
}

#endif // OMTALK_OMTALKOPS_TD_
