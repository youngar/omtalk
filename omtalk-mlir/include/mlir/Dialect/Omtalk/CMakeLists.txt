
# This must be in the same directory as the tablegen file
set(LLVM_TARGET_DEFINITIONS OmtalkOps.td)
omtalk_tablegen(OmtalkOps.h.inc -gen-op-decls)
omtalk_tablegen(OmtalkOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(omtalk-mlir-tblgen)
