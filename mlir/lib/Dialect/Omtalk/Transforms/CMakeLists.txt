set(LLVM_TARGET_DEFINITIONS Canonicalize.td)
mlir_tablegen(Canonicalize.inc -gen-rewriters "-I${PROJECT_SOURCE_DIR}/omtalk-mlir/include")
add_public_tablegen_target(omtalk-canonicalize-tblgen)
