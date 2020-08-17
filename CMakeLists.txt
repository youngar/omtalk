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

# High level Omtalk components
option(OMTALK_COMPILER "Build with compiler support")
option(OMTALK_CORE "Build the Omtalk core libraries.  This is the main language VM. Requires all other components to be built.")
option(OMTALK_GC "Build the Omtalk garbage collector")
option(OMTALK_OM "Build the Omtalk Om object model. Requires OMTALK_COMPILER and OMTALK_GC")
option(OMTALK_PARSER "Build the Omtalk SOM parser.  A parsing library for the Simple Object Machine language.")

# Build customization
option(OMTALK_ASAN "Build with clang address sanitizer enabled.")
option(OMTALK_LLD "Use the LLVM linker ld.lld")
option(OMTALK_LLVM "Build against the LLVM submodule")
option(OMTALK_RTTI "Build with RTTI support.")
option(OMTALK_SPLIT_DEBUG "Split debug information for faster link times")
option(OMTALK_TESTS "Build and run the tests")
option(OMTALK_UBSAN "Build with clang undefined behaviour sanitizer.")
option(OMTALK_WARNINGS "Build with extra warning enabled")

set(OMTALK_PATH ${omtalk_SOURCE_DIR}/external/SOM/Smalltalk CACHE STRING "The core library class path")

include(HandleOmtalkOptions)

###
### LLVM integration
###

if(OMTALK_LLVM)
# TODO: Including these files here causes LLVM to change how everything
# is built across all of omtalk
include(BuildLLVM)
endif()

###
### Subdirectories
###

# tools/omtalk-tblgen must be added from this directory. This is because of the
# way tablegen cmake helpers modify the parent scope.
if(OMTALK_OM)
  add_subdirectory(tools/omtalk-tblgen)
endif()

if(OMTALK_COMPILER)
  add_subdirectory(compiler)
endif()

if(OMTALK_CORE)
  add_subdirectory(core)
endif()

add_subdirectory(external)

if(OMTALK_GC)
  add_subdirectory(gc)
endif()

if(OMTALK_PARSER)
add_subdirectory(parser)
endif()

if(OMTALK_TEST)
add_subdirectory(test)
endif()

add_subdirectory(tools)

add_subdirectory(util)
