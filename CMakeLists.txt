cmake_minimum_required(VERSION 3.10)

project(omtalk
	LANGUAGES C CXX ASM_NASM
	VERSION 0.1)

# Global

enable_testing()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS true)
set(CMAKE_BUILD_TYPE Debug)

# Configuration Options

# set(OMTALK_LLVM_OPTIONS "" CACHE "Options to pass to LLVM")
set(OMTALK_LLVM_OPTIONS "")

set(OMTALK_ASAN OFF CACHE BOOL "Build with clang address sanitizer enabled.")
if(OMTALK_ASAN)
	add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
	link_libraries(-fsanitize=address)
endif()

set(OMTALK_UBSAN OFF CACHE BOOL "Build with clang undefined behaviour sanitizer.")
if(OMTALK_UBSAN)
	add_compile_options(-fsanitize=undefined)
	link_libraries(-fsanitize=undefined)
endif()

set(OMTALK_RTTI OFF CACHE BOOL "Build with RTTI support.")
if(NOT OMTALK_RTTI)
	add_compile_options(-fno-rtti)
	list(APPEND OMTALK_LLVM_OPTIONS -DLLVM_ENABLE_RTTI=off)
else()
	list(APPEND OMTALK_LLVM_OPTIONS -DLLVM_ENABLE_RTTI=on)
endif()

# LLVM

include(BuildLLVM)
build_llvm(external/llvm-project
	OPTIONS
		-DLLVM_ENABLE_PROJECTS=mlir
		-DLLVM_PARALLEL_LINK_JOBS=2
		${OMTALK_LLVM_OPTIONS})

find_package(MLIR REQUIRED PATHS ${CMAKE_MODULE_PATH} NO_DEFAULT_PATH)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Subdirectories

# add_subdirectory(external)
add_subdirectory(mlir)
