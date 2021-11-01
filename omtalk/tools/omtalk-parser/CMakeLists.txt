get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(omtalk-parser-bin
	omtalk-parser.cpp
)

set_target_properties(omtalk-parser-bin
	PROPERTIES
		OUTPUT_NAME omtalk-parser
)

llvm_update_compile_flags(omtalk-parser-bin)

target_link_libraries(omtalk-parser-bin
	PRIVATE
		omtalk-core
		omtalk-parser
		omtalk-ir
		LLVMCore
		LLVMSupport
		${dialect_libs}
		# ${conversion_libs}
)
