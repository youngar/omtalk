get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(omtalk-bin
	omtalk.cpp
)

set_target_properties(omtalk-bin
	PROPERTIES
		OUTPUT_NAME omtalk
)

llvm_update_compile_flags(omtalk-bin)

target_link_libraries(omtalk-bin
	PRIVATE
		omtalk-core
		omtalk-parser
		omtalk-ir
		LLVMCore
		LLVMSupport
		${dialect_libs}
		# ${conversion_libs}
)
