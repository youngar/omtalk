#ifndef OMTALK_OPS_TD_
#define OMTALK_OPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Analysis/CallInterfaces.td"

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

// Provide a definition for the Toy StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef.
def Omtalk_BoxType :
    Type<CPred<"$_self.isa<::omtalk::BoxType>()">, "Omtalk Boxed Type">;

def Omtalk_BoxIntType :
    Type<CPred<"$_self.isa<::omtalk::BoxIntType>()">, "Omtalk Boxed Primitive Integer Type">;

def Omtalk_BoxRefType :
    Type<CPred<"$_self.isa<::omtalk::BoxRefType>()">, "Omtalk Boxed Reference Type">;

// Provide a definition of the types that are used within the Toy dialect.
def Omtalk_Type : AnyTypeOf<[Omtalk_BoxType, Omtalk_BoxIntType, Omtalk_BoxRefType]>;

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

def ConstantIntOp : Omtalk_Op<"constant_int"> {
  let summary = "constant_int";
  let description = [{
      A constant boxed integer
    }];
  let arguments = (ins I64Attr:$value);
  let results = (outs Omtalk_BoxIntType);
}

def SendOp : Omtalk_Op<"send",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "Send symbol to receiver";
  let description = [{
    The "send" operation sends the message to the receiver.
  }];

  let arguments = (ins Omtalk_Type:$recv, FlatSymbolRefAttr:$message, Variadic<Omtalk_Type>:$inputs);
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

#endif // OMTALK_OPS_TD_
