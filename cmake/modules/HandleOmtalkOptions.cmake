###
### Sanitizer Support
###

if(OMTALK_ASAN)
	add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
	link_libraries(-fsanitize=address)
endif()

if(OMTALK_UBSAN)
	add_compile_options(-fsanitize=undefined)
	link_libraries(-fsanitize=undefined)
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
### Split Debug Information
###

if(OMTALK_SPLIT_DEBUG)
	add_compile_options(-gsplit-dwarf)
	add_link_options(-Wl,--gdb-index)
	list(APPEND OMTALK_LLVM_OPTIONS -DLLVM_USE_SPLIT_DWARF=on)
endif()