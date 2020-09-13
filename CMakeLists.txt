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

include(CMakeFindDependencyMacro)
find_dependency(Threads)

###
### Configuration
###

# High level Omtalk components
option(OMTALK_CCACHE "Build using ccache as a compiler launcher")
option(OMTALK_CHECKS "Enable runtime assertions")
option(OMTALK_CHECKS_EXPENSIVE "Enable expensive runtime assertions")
option(OMTALK_COMP_COMPILER "Build with compiler support")
option(OMTALK_COMP_CORE "Build the Omtalk core libraries.  This is the main language VM. Requires all other components to be built.")
option(OMTALK_COMP_GC "Build the Omtalk garbage collector")
option(OMTALK_COMP_OM "Build the Omtalk Om object model. Requires OMTALK_COMP_COMPILER and OMTALK_COMP_GC")
option(OMTALK_COMP_PARSER "Build the Omtalk SOM parser.  A parsing library for the Simple Object Machine language.")
option(OMTALK_LLD "Use the LLVM linker ld.lld")
option(OMTALK_LLVM "Build against the LLVM submodule")
option(OMTALK_RTTI "Build with RTTI support.")
option(OMTALK_SAN_ASAN "Build with clang address sanitizer enabled.")
option(OMTALK_SAN_TSAN " Build with clang thread sanitizer enabled")
option(OMTALK_SAN_UBSAN "Build with clang undefined behaviour sanitizer.")
option(OMTALK_SPLIT_DEBUG "Split debug information for faster link times")
option(OMTALK_STATIC_CLANG_TIDY "Build with the clang-tidy static analyzer")
option(OMTALK_STATIC_IWYU "Build with the include-what-you-use static analyzer")
option(OMTALK_STATIC_LWYU "Build with the link-what-you-use static analyzer")
option(OMTALK_TESTS "Build and run the tests")
option(OMTALK_WARNINGS "Build with extra warning enabled")
option(OMTALK_WARNINGS_AS_ERRORS "Treat compiler warnings as errors")

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
if(OMTALK_COMP_OM)
  add_subdirectory(tools/omtalk-tblgen)
endif()

if(OMTALK_COMP_COMPILER)
  add_subdirectory(compiler)
endif()

if(OMTALK_COMP_CORE)
  add_subdirectory(core)
endif()

add_subdirectory(external)

if(OMTALK_COMP_GC)
  add_subdirectory(gc)
endif()

if(OMTALK_COMP_PARSER)
add_subdirectory(parser)
endif()

if(OMTALK_COMP_TEST)
add_subdirectory(test)
endif()

add_subdirectory(tools)

add_subdirectory(util)
