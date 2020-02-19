
#ifndef OMTALK_CANONICALIZE_TD_
#define OMTALK_CANONICALIZE_TD_

include "mlir/Dialect/Omtalk/OmtalkOps.td"

//
// Integer Canonicalization
//

def ReplaceSendInt : Pat<
    (SendOp Omtalk_BoxIntType:$recv, $message, $args),
    (SendIntOp $recv, $message, $args)>;

#endif // OMTALK_CANONICALIZE_TD_
