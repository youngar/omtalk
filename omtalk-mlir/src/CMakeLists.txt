set(LLVM_TARGET_DEFINITIONS canonicalize.td)
omtalk_tablegen(canonicalize.inc -gen-rewriters "-I${CMAKE_CURRENT_SOURCE_DIR}/../include")
add_public_tablegen_target(omtalk-canonicalize-tblgen)