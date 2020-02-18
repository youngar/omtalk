
#ifndef OMTALK_CANONICALIZE_TD_
#define OMTALK_CANONICALIZE_TD_

include "omtalk/ops.td"

//
// Integer Canonicalization
//

def ReplaceSendInt : Pat<
    (SendOp Omtalk_BoxIntType:$recv, $message, $args),
    (SendIntOp $recv, $message, $args)>;

#endif // OMTALK_CANONICALIZE_TD_