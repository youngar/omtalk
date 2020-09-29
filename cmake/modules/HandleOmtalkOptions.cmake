if(HANDLE_OMTALK_OPTIONS_)
	return()
endif()
set(HANDLE_OMTALK_OPTIONS_ TRUE)

set(LLVM_ENABLE_PROJECTS "" CACHE STRING "LLVM projects to enable")
set(OMTALK_COMPILE_OPTIONS  "" CACHE STRING "Options passed to the compiler")
set(OMTALK_LINK_OPTIONS "" CACHE STRING "Options pass to the linker")
set(OMTALK_LLVM_OPTIONS "" CACHE STRING "Options passed to LLVM")

###
### LLVM Configuration
###

list(APPEND OMTALK_LLVM_OPTIONS
	-DLLVM_PARALLEL_LINK_JOBS=2
	-DLLVM_BUILD_EXAMPLES=on
)

# TODO this is interpreted by HandleLLVMOptions.cmake
set(LLVM_PARALLEL_LINK_JOBS 2)

list(APPEND LLVM_ENABLE_PROJECTS
	mlir
)

list(APPEND OMTALK_LLVM_OPTIONS
	-DLLVM_TARGETS_TO_BUILD=X86
	-DLLVM_OPTIMIZED_TABLEGEN=true
	-DLLVM_CCACHE_BUILD=true
)

###
### Warnings and Errors
###

if(OMTALK_WARNINGS)
	list(APPEND OMTALK_COMPILE_OPTIONS 
		$<$<COMPILE_LANGUAGE:CXX,C>:-Wall>
		$<$<COMPILE_LANGUAGE:CXX,C>:-Wno-unused-parameter>
		$<$<COMPILE_LANGUAGE:CXX,C>:-Wno-unused-function>
	)
endif()

if(OMTALK_WARNINGS_AS_ERRORS)
	list(APPEND OMTALK_COMPILE_OPTIONS 
		$<$<COMPILE_LANGUAGE:CXX,C>:-Werror>
	)
endif()

set(LLVM_ENABLE_WARNINGS off)
set(LLVM_ENABLE_PEDANTIC off)
set(LLVM_ENABLE_WERROR off)

###
### Sanitizer Support
###

if(OMTALK_SAN_ASAN)
	list(APPEND OMTALK_COMPILE_OPTIONS 
		$<$<COMPILE_LANGUAGE:CXX,C>:-fsanitize=address -fno-omit-frame-pointer>
	)
	list(APPEND OMTALK_LINK_OPTIONS
		-fsanitize=address
	)
endif()

if(OMTALK_SAN_TSAN)
	list(APPEND OMTALK_COMPILE_OPTIONS
		$<$<COMPILE_LANGUAGE:CXX,C>:-fsanitize=thread>
	)
	list(APPEND OMTALK_LINK_OPTIONS
		-fsanitize=thread
	)
endif()

if(OMTALK_SAN_UBSAN)
	list(APPEND OMTALK_COMPILE_OPTIONS
		$<$<COMPILE_LANGUAGE:CXX,C>:-fsanitize=undefined>
	)
	list(APPEND OMTALK_LINK_OPTIONS
		-fsanitize=undefined
	)
endif()

if(OMTALK_SAN_THREADSAFETY)
	list(APPEND OMTALK_COMPILE_OPTIONS
		$<$<COMPILE_LANGUAGE:CXX,C>:-Wthread-safety>
	)
	# TODO: Not all of the standard library has been annoted, only certain
	# pieces.  This creates false errors which are impossible to fix.  In
	# particular we need std::unique_lock to receive annotations.
	#
	# list(APPEND OMTALK_COMPILE_DEFINITIONS
	# 	_LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS
	# )
endif()

###
### RTTI and Exceptions
###

if(NOT OMTALK_RTTI)
	list(APPEND OMTALK_COMPILE_OPTIONS
		$<$<COMPILE_LANGUAGE:CXX,C>:-fno-rtti>
	)
	list(APPEND OMTALK_LLVM_OPTIONS -DLLVM_ENABLE_RTTI=off)
else()
	list(APPEND OMTALK_LLVM_OPTIONS -DLLVM_ENABLE_RTTI=on)
endif()

###
### Linker support
###

if(OMTALK_LLD)
	list(APPEND OMTALK_LLVM_OPTIONS -DLLVM_ENABLE_LLD=on)
	list(APPEND LLVM_ENABLE_PROJECTS lld)
	# TODO is this line needed?
	set(LLVM_ENABLE_LLD on)
	add_link_options(-fuse-ld=lld)
endif()

###
### Split Debug Information
###

if(OMTALK_SPLIT_DEBUG)
	add_compile_options(
		$<$<COMPILE_LANGUAGE:CXX,C>:-gsplit-dwarf>
	)
	add_link_options(
		$<$<AND:$<COMPILE_LANGUAGE:C>,$<C_COMPILER_ID:GNU>>:-Wl,--gdb-index>
		$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wl,--gdb-index>
	)
	list(APPEND OMTALK_LLVM_OPTIONS
		-DLLVM_USE_SPLIT_DWARF=on
	)
endif()
