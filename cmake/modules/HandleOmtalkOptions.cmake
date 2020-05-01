
set(OMTALK_LLVM_OPTIONS "" CACHE STRING "")
set(LLVM_ENABLE_PROJECTS "" CACHE STRING "")

list(APPEND OMTALK_LLVM_OPTIONS
	-DLLVM_PARALLEL_LINK_JOBS=2
	-DLLVM_BUILD_EXAMPLES=on
)

# TODO this is interpreted by HandleLLVMOptions.cmake
set(LLVM_PARALLEL_LINK_JOBS 2)

list(APPEND LLVM_ENABLE_PROJECTS
	mlir
)

###
### Sanitizer Support
###

if(OMTALK_ASAN)
	add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
	add_link_options(-fsanitize=address)
endif()

if(OMTALK_UBSAN)
	add_compile_options(-fsanitize=undefined)
	add_link_options(-fsanitize=undefined)
endif()

###
### RTTI and Exceptions
###

if(NOT OMTALK_RTTI)
	add_compile_options(-fno-rtti)
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
	add_compile_options(-gsplit-dwarf)
	add_link_options(-Wl,--gdb-index)
	list(APPEND OMTALK_LLVM_OPTIONS -DLLVM_USE_SPLIT_DWARF=on)
endif()
