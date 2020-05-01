cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(omtalk
	LANGUAGES C CXX ASM_NASM
	VERSION 0.1
)

###
### Global
###

enable_testing()

###
### Modules
###

list(APPEND CMAKE_MODULE_PATH ${omtalk_SOURCE_DIR}/cmake/modules)
include(AddOmtalk)

###
### Configuration
###

option(OMTALK_ASAN "Build with clang address sanitizer enabled.")
option(OMTALK_LLD "Use the LLVM linker ld.lld")
option(OMTALK_RTTI "Build with RTTI support.")
option(OMTALK_SPLIT_DEBUG "Split debug information for faster link times")
option(OMTALK_UBSAN "Build with clang undefined behaviour sanitizer.")

include(HandleOmtalkOptions)

###
### LLVM integration
###

include(BuildLLVM)

###
### Subdirectories
###

add_subdirectory(external)
add_subdirectory(omtalk-gc)
add_subdirectory(omtalk-mlir)
add_subdirectory(omtalk-parser)
add_subdirectory(omtalk)
add_subdirectory(test)
add_subdirectory(tools)