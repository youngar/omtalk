cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 17)

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
include(OmOptions)
include(OmPlatform)
include(CMakeFindDependencyMacro)

find_dependency(Threads)

###
### Options
###

om_add_option_group(OMTALK_OPTIONS)

## Language Edition + Features

om_add_option(OMTALK_RTTI OMTALK_OPTIONS
	DOC "Build with RTTI support."
)

om_add_option(OMTALK_WARNINGS OMTALK_OPTIONS
	DOC "Build with extra warning enabled"
	DEFAULT CMAKE_BUILD_TYPE STREQUAL "Debug"
)

om_add_option(OMTALK_WARNINGS_AS_ERRORS OMTALK_OPTIONS
	DOC "Treat compiler warnings as errors"
	DEFAULT CMAKE_BUILD_TYPE STREQUAL "Debug"
)

## High Level Components

om_add_option(OMTALK_COMP_COMPILER OMTALK_OPTIONS
	DOC "Build with compiler support"
)

om_add_option(OMTALK_COMP_CORE OMTALK_OPTIONS
	DOC "Build the Omtalk core libraries.  This is the main language VM."
	REQUIRES
		OMTALK_COMP_COMPILER
		OMTALK_COMP_GC
		OMTALK_COMP_OM
		OMTALK_COMP_PARSER
)

om_add_option(OMTALK_COMP_GC OMTALK_OPTIONS
	DOC "Build the Omtalk garbage collector"
)

om_add_option(OMTALK_COMP_OM OMTALK_OPTIONS
	DOC "Build the Omtalk Om object model."
	REQUIRES
		OMTALK_COMP_GC
)

om_add_option(OMTALK_COMP_PARSER OMTALK_OPTIONS
	DOC "Build the Omtalk SOM parser.  A parsing library for the Simple Object Machine language."
)

om_add_option(OMTALK_TESTS OMTALK_OPTIONS
	DOC "Build and run the tests"
	DEFAULT CMAKE_BUILD_TYPE STREQUAL "Debug"
)

om_add_option(OMTALK_LLVM OMTALK_OPTIONS
	DOC "Build with LLVM"
)
om_add_option(OMTALK_BUILD_LLVM OMTALK_OPTIONS
	DOC "Build the included LLVM submodule"
	REQUIRES OMTALK_LLVM
)

set(OMTALK_PATH ${omtalk_SOURCE_DIR}/external/SOM/Smalltalk CACHE STRING "The core library class path")

## Checks and Assertions

om_add_option(OMTALK_CHECKS OMTALK_OPTIONS
	DOC "Enable runtime assertions"
	DEFAULT CMAKE_BUILD_TYPE STREQUAL "Debug"
)

om_add_option(OMTALK_CHECKS_EXPENSIVE OMTALK_OPTIONS
	DOC "Enable expensive runtime assertions"
	REQUIRES OMTALK_CHECKS
)

##  LLVM-Specific Tooling

om_add_option(OMTALK_LLVM_TOOLCHAIN OMTALK_OPTIONS
	DOC "Enable if we are using the clang/llvm toolchain"
	DEFAULT CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang|ARMClang)$"
)

om_add_option(OMTALK_SAN_ASAN OMTALK_OPTIONS
	DOC "Build with clang address sanitizer enabled."
	REQUIRES OM_HAVE_ASAN
	CONFLICTS OMTALK_SAN_TSAN
)

om_add_option(OMTALK_SAN_TSAN OMTALK_OPTIONS
	DOC "Build with clang thread sanitizer enabled"
	REQUIRES OM_HAVE_TSAN
	CONFLICTS OMTALK_SAN_ASAN
)

om_add_option(OMTALK_SAN_THREADSAFETY OMTALK_OPTIONS
	DOC "Build with clang thread safety annotations enabled."
)

om_add_option(OMTALK_SAN_UBSAN OMTALK_OPTIONS
	DOC "Build with clang undefined behaviour sanitizer."
	REQUIRES OM_HAVE_UBSAN
)

om_add_option(OMTALK_SPLIT_DEBUG OMTALK_OPTIONS
	DOC "Split debug information for faster link times."
	DEFAULT
		( CMAKE_BUILD_TYPE STREQUAL "Debug" ) OR
		( CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" )
	REQUIRES OMTALK_LLVM_TOOLCHAIN
)

find_program(OMTALK_TOOL_LLD ld.lld)

om_add_option(OMTALK_LLD OMTALK_OPTIONS
	DOC "Use the LLVM linker ld.lld"
	DEFAULT  OMTALK_TOOL_LLD
	REQUIRES OMTALK_TOOL_LLD
)

## Misc Tooling and Linters

find_program(OMTALK_TOOL_CCACHE ccache)

om_add_option(OMTALK_CCACHE OMTALK_OPTIONS
	DOC "Build using ccache as a compiler launcher"
	DEFAULT  OMTALK_TOOL_CCACHE
	REQUIRES OMTALK_TOOL_CCACHE
)

find_program(OMTALK_TOOL_CLANG_TIDY clang-tidy)

om_add_option(OMTALK_STATIC_CLANG_TIDY OMTALK_OPTIONS
	DOC "Build with the clang-tidy static analyzer"
	DEFAULT  OMTALK_TOOL_CLANG_TIDY
	REQUIRES OMTALK_TOOL_CLANG_TIDY
)

find_program(OMTALK_IWYU_EXECUTABLE include-what-you-use)

om_add_option(OMTALK_STATIC_IWYU OMTALK_OPTIONS
	DOC "Build with the include-what-you-use static analyzer"
	DEFAULT  OMTALK_IWYU_EXECUTABLE
	REQUIRES OMTALK_IWYU_EXECUTABLE
)

om_add_option(OMTALK_STATIC_LWYU OMTALK_OPTIONS
	DOC "Build with the link-what-you-use static analyzer"
)

om_validate_option_group(OMTALK_OPTIONS)

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
