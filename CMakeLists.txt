cmake_minimum_required(VERSION 3.2)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

project(gencomp
	LANGUAGES C CXX ASM_NASM
	VERSION 0.1
)

# Global settings

add_compile_options(
	-fno-rtti
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS true)
set(CMAKE_BUILD_TYPE Debug)
set(LLVM_ENABLE_PROJECTS mlir)
set(LLVM_BUILD_EXAMPLES ON)
set(LLVM_TARGETS_TO_BUILD host)
set(LLVM_ENABLE_ASSERTIONS ON)

# cmake -G Ninja ../llvm \
#    -DLLVM_ENABLE_PROJECTS=mlir \
#    -DLLVM_BUILD_EXAMPLES=ON \
#    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DLLVM_ENABLE_ASSERTIONS=ON \

enable_testing()

# Optional debug and testing features

set(ASAN OFF CACHE BOOL "Build b9 with clang address sanitizer enabled.")
if(ASAN)
	add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
	link_libraries(-fsanitize=address)
endif(ASAN)

set(UBSAN OFF CACHE BOOL "Build b9 with clang undefined behaviour sanitizer.")
if(UBSAN)
	add_compile_options(-fsanitize=undefined)
	link_libraries(-fsanitize=undefined)
endif(UBSAN)

# Subdirectories

add_subdirectory(external)

add_subdirectory(omtalk-mlir)
