
# This must be in the same directory as ops.td.
set(LLVM_TARGET_DEFINITIONS ops.td)
omtalk_tablegen(ops.h.inc -gen-op-decls)
omtalk_tablegen(ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(omtalk-mlir-tblgen)
