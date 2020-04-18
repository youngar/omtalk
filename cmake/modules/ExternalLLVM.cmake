# if(OMTALK_LLVM_)
#     return()
# endif()
# set(OMTALK_LLVM_ true)

include(ExternalProject)

set(LLVM_PROJECT_REPO https://github.com/llvm/llvm-project.git)
set(LLVM_PROJECT_TAG master)

# cmake -G Ninja ../llvm \
#    -DLLVM_ENABLE_PROJECTS=mlir \
#    -DLLVM_BUILD_EXAMPLES=ON \
#    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DLLVM_ENABLE_ASSERTIONS=ON

ExternalProject_Add(
	llvm
	GIT_REPOSITORY ${LLVM_PROJECT_REPO}
	GIT_TAG ${LLVM_PROJECT_TAG}
	GIT_SHALLOW true
	SOURCE_SUBDIR llvm
	CMAKE_ARGS
		-DLLVM_ENABLE_PROJECTS=mlir
		-DLLVM_TARGETS_TO_BUILD=host
		-DCMAKE_BUILD_TYPE=Release
		-DLLVM_ENABLE_ASSERTIONS=ON
	UPDATE_COMMAND ""
	INSTALL_COMMAND ""
)

ExternalProject_Get_Property(llvm SOURCE_DIR BINARY_DIR)

set(LLVM_PROJECT_SOURCE_DIR ${SOURCE_DIR})
set(LLVM_PROJECT_BINARY_DIR ${BINARY_DIR})
set(LLVM_DIR "${LLVM_PROJECT_BINARY_DIR}/lib/cmake/llvm")

set(MLIR_INCLUDE_DIRS 
	"${LLVM_PROJECT_SOURCE_DIR}/mlir/include"
	"${LLVM_PROJECT_BINARY_DIR}/tools/mlir/include")

list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}/llvm/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")

message("LLVM_PROJECT_SOURCE_DIR = \"${LLVM_PROJECT_SOURCE_DIR}\"")
message("LLVM_PROJECT_BINARY_DIR = \"${LLVM_PROJECT_BINARY_DIR}\"")
message("CMAKE_MODULE_PATH = \"${CMAKE_MODULE_PATH}\"")

# /home/aryoung/wsp/compiler/build/llvm-prefix/src/llvm-build/cmake/modules/CMakeFiles/LLVMConfig.cmake
# /home/aryoung/wsp/compiler/build/llvm-prefix/src/llvm/llvm/cmake/modules/AddLLVM.cmake

# include(AddLLVM)
find_package(LLVM REQUIRED CONFIG)

message("MLIR_INCLUDE_DIRS = \"${MLIR_INCLUDE_DIRS}\"")
message("LLVM_INCLUDE_DIRS = \"${LLVM_INCLUDE_DIRS}\"")
message("LLVM_DEFINITIONS = \"${LLVM_DEFINITIONS}\"")

# llvm_map_components_to_libnames(llvm_libs support core irreader)
# message("llvm_libs = \"${llvm_libs}\"")

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using modules in: ${LLVM_DIR}")
message(STATUS "LLVM RTTI is ${LLVM_ENABLE_RTTI}")

# TODO do we need this?
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${})
add_definitions(${LLVM_DEFINITIONS})
