get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRStandalone
        )
add_llvm_executable(omtalk-opt omtalk-opt.cpp)

llvm_update_compile_flags(omtalk-opt)
target_link_libraries(omtalk-opt PRIVATE ${LIBS})
