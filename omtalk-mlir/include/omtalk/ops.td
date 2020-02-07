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
    Type<CPred<"$_self.isa<::omtalk::BoxType>()">, "Omtalk Box Type">;

// Provide a definition of the types that are used within the Toy dialect.
def Omtalk_Type : AnyTypeOf<[Omtalk_BoxType]>;

//
// Operations
//

def ConstantOp : Omtalk_Op<"constant", [NoSideEffect]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute.
    }];
  let arguments = (ins I64Attr:$value);
  let results = (outs Omtalk_BoxType);
  // let verifier = [{ return ::verify(*this); }];
}

def IAddOp : Omtalk_Op<"iadd"> {
  let summary = "Integer add operator";
  let description = [{
    The "IAdd" operation adds two integer constants together.
  }];

  let arguments = (ins Omtalk_BoxType:$recv, Omtalk_BoxType:$rhs);
  let results = (outs Omtalk_BoxType);

  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs">
  ];
}

def StaticSendOp : Omtalk_Op<"static_send",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "Send symbol to receiver";
  let description = [{
    The "send" operation sends the message to the receiver.
  }];

  let arguments = (ins FlatSymbolRefAttr:$message, Variadic<Omtalk_BoxType>:$inputs);
  let results = (outs Omtalk_BoxType);
}

def ReturnOp : Omtalk_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  let summary = "return to the sender";
  let description = [{
    The "return" operation represents a return from the sender to the receiver.
    The receiver always returns a value for a message expression.
  }];

  let arguments = (ins Omtalk_BoxType:$value);
}

#endif // OMTALK_OPS_TD_
