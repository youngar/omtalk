set(LLVM_TARGET_DEFINITIONS Omtalk.td)
mlir_tablegen(OmtalkOps.h.inc -gen-op-decls)
mlir_tablegen(OmtalkOps.cpp.inc -gen-op-defs)
mlir_tablegen(OmtalkTypes.h.inc -gen-typedef-decls)
mlir_tablegen(OmtalkTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(OmtalkDialect.h.inc -gen-dialect-decls -dialect=omtalk)
mlir_tablegen(OmtalkDialect.cpp.inc -gen-dialect-defs -dialect=omtalk)
add_public_tablegen_target(omtalk-ir-incgen)

add_mlir_doc(OmtalkDialect OmtalkDialect Omtalk/ -gen-dialect-doc)
add_mlir_doc(OmtalkOps OmtalkOps Omtalk/ -gen-op-doc)
