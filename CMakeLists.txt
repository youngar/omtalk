cmake_minimum_required(VERSION 3.13)

# This is needed when including LLVM.
# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)

###
### Omtalk
###

project(omtalk
  LANGUAGES C CXX 
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
include(OmOptions)
include(OmPlatform)
include(CMakeFindDependencyMacro)

find_dependency(Threads)

###
### Options
###

om_add_option_group(OM_OPTIONS)

## Language Edition + Features

om_add_option(OMTALK_RTTI OM_OPTIONS
  DOC "Build with RTTI support."
)

om_add_option(OMTALK_WARNINGS OM_OPTIONS
  DOC "Build with extra warning enabled"
  DEFAULT CMAKE_BUILD_TYPE STREQUAL "Debug"
)

om_add_option(OMTALK_WARNINGS_AS_ERRORS OM_OPTIONS
  DOC "Treat compiler warnings as errors"
  DEFAULT CMAKE_BUILD_TYPE STREQUAL "Debug"
)

## Global Settings

om_add_option(OM_TESTS OM_OPTIONS
  DOC "Build and run the tests"
  DEFAULT PROJECT_IS_TOP_LEVEL
)

om_add_option(OM_BENCHMARKS OM_OPTIONS
  DOC "Build and run the benchmarks"
  DEFAULT PROJECT_IS_TOP_LEVEL
)

om_add_option(OMTALK_LLVM OM_OPTIONS
  DOC "Build with LLVM"
)

om_add_option(OM_LIT_TESTS OM_OPTIONS
  DOC "Build and run the tests"
  DEFAULT OMTALK_LLVM
)

om_add_option(OMTALK_BUILD_LLVM OM_OPTIONS
  DOC "Build the included LLVM submodule"
  REQUIRES OMTALK_LLVM
)

## Checks and Assertions

om_add_option(OMTALK_CHECKS OM_OPTIONS
  DOC "Enable runtime assertions"
  DEFAULT CMAKE_BUILD_TYPE STREQUAL "Debug"
)

om_add_option(OMTALK_CHECKS_EXPENSIVE OM_OPTIONS
  DOC "Enable expensive runtime assertions"
  REQUIRES OMTALK_CHECKS
)

##  LLVM-Specific Tooling

om_add_option(OMTALK_LLVM_TOOLCHAIN OM_OPTIONS
  DOC "Enable if we are using the clang/llvm toolchain"
  DEFAULT CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang|ARMClang)$"
)

om_add_option(OMTALK_SAN_ASAN OM_OPTIONS
  DOC "Build with clang address sanitizer enabled."
  REQUIRES OM_HAVE_ASAN
  CONFLICTS OMTALK_SAN_TSAN
)

om_add_option(OMTALK_SAN_TSAN OM_OPTIONS
  DOC "Build with clang thread sanitizer enabled"
  REQUIRES OM_HAVE_TSAN
  CONFLICTS OMTALK_SAN_ASAN
)

om_add_option(OMTALK_SAN_THREADSAFETY OM_OPTIONS
  DOC "Build with clang thread safety annotations enabled."
  DEFAULT ON
)

om_add_option(OMTALK_SAN_UBSAN OM_OPTIONS
  DOC "Build with clang undefined behaviour sanitizer."
  REQUIRES OM_HAVE_UBSAN
)

om_add_option(OMTALK_SPLIT_DEBUG OM_OPTIONS
  DOC "Split debug information for faster link times."
  DEFAULT
    ( CMAKE_BUILD_TYPE STREQUAL "Debug" ) OR
    ( CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" )
  REQUIRES OMTALK_LLVM_TOOLCHAIN
)

find_program(OMTALK_TOOL_LLD ld.lld)

om_add_option(OMTALK_LLD OM_OPTIONS
  DOC "Use the LLVM linker ld.lld"
  DEFAULT  OFF
  REQUIRES OMTALK_TOOL_LLD
)

## Misc Tooling and Linters

find_program(OMTALK_TOOL_CCACHE ccache)

om_add_option(OMTALK_CCACHE OM_OPTIONS
  DOC "Build using ccache as a compiler launcher"
  DEFAULT  OMTALK_TOOL_CCACHE
  REQUIRES OMTALK_TOOL_CCACHE
)

find_program(OMTALK_TOOL_CLANG_TIDY clang-tidy)

om_add_option(OMTALK_STATIC_CLANG_TIDY OM_OPTIONS
  DOC "Build with the clang-tidy static analyzer"
  DEFAULT  OMTALK_TOOL_CLANG_TIDY
  REQUIRES OMTALK_TOOL_CLANG_TIDY
)

find_program(OMTALK_IWYU_EXECUTABLE include-what-you-use)

om_add_option(OMTALK_STATIC_IWYU OM_OPTIONS
  DOC "Build with the include-what-you-use static analyzer"
  DEFAULT  OMTALK_IWYU_EXECUTABLE
  REQUIRES OMTALK_IWYU_EXECUTABLE
)

om_add_option(OMTALK_STATIC_LWYU OM_OPTIONS
  DOC "Build with the link-what-you-use static analyzer"
)

## High Level Components

# All component flags are disabled by default, and should be enabled using 
# CMake caches.

## AB

om_add_option(OM_COMP_AB OM_OPTIONS
  DOC "Build the AB utility library"
)

## Om

om_add_option(OM_COMP_OM_OM OM_OPTIONS
  DOC "Build the OM object model"
  REQUIRES
    OM_COMP_AB
)

om_add_option(OM_COMP_OM_GC OM_OPTIONS
  DOC "Build the Omtalk garbage collector"
  REQUIRES
    OM_COMP_OM
)

om_add_option(OM_COMP_OM_OM OM_OPTIONS
  DOC "Build the Omtalk Om object model."
  REQUIRES
    OM_COMP_OM_GC
)

## Omtalk

om_add_option(OM_COMP_OMTALK OM_OPTIONS
  DOC "Build the omtalk language"
  REQUIRES
    OM_COMP_OM_OM
)

om_add_option(OM_COMP_OMTALK_COMPILER OM_OPTIONS
  DOC "Build with compiler support"
  REQUIRES
    OM_COMP_OMTALK
)

om_add_option(OM_COMP_OMTALK_CORE OM_OPTIONS
  DOC "Build the Omtalk core libraries.  This is the main language VM."
  REQUIRES
    OM_COMP_OM_GC
    OM_COMP_OM_OM
    OM_COMP_OMTALK
    OM_COMP_OMTALK_COMPILER
    OM_COMP_OMTALK_PARSER
)

om_add_option(OM_COMP_OMTALK_PARSER OM_OPTIONS
  DOC "Build the Omtalk SOM parser.  A parsing library for the Simple Object Machine language."
)

set(OMTALK_PATH ${omtalk_SOURCE_DIR}/external/SOM/Smalltalk CACHE STRING "The core library class path")

om_validate_option_group(OM_OPTIONS)

###
### Option Handling
###

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

if(OM_COMP_AB)
  add_subdirectory(ab)
endif()

add_subdirectory(external)

if(OM_COMP_OM_OM)
  add_subdirectory(om)
endif()

if(OM_COMP_OMTALK)
  add_subdirectory(omtalk)
endif()
