cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()

###
### Omtalk
###

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
option(OMTALK_WARNINGS "Build with extra warning enabled")

set(OMTALK_PATH ${omtalk_SOURCE_DIR}/external/SOM/Smalltalk CACHE STRING "The core library class path")

include(HandleOmtalkOptions)

###
### LLVM integration
###

# TODO: Including these files here causes LLVM to change how everything
# is built across all of omtalk
include(BuildLLVM)

###
### Subdirectories
###

add_subdirectory(compiler)
add_subdirectory(core)
add_subdirectory(external)
add_subdirectory(gc)
add_subdirectory(parser)
add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(util)
