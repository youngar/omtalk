###
### LLVM CMake Integration
###

set(OMTALK_LLVM_SOURCE_DIR "${omtalk_SOURCE_DIR}/external/llvm-project")
set(OMTALK_LLVM_BINARY_DIR "${omtalk_BINARY_DIR}/external/llvm-project")

file(MAKE_DIRECTORY "${OMTALK_LLVM_BINARY_DIR}")

list(APPEND OMTALK_LLVM_OPTIONS
	-DLLVM_ENABLE_PROJECTS=mlir
	-DLLVM_PARALLEL_LINK_JOBS=2
	-DLLVM_BUILD_EXAMPLES=on
)

execute_process(
	COMMAND "${CMAKE_COMMAND}" "${OMTALK_LLVM_SOURCE_DIR}/llvm" -G "${CMAKE_GENERATOR}" ${OMTALK_LLVM_OPTIONS}
	WORKING_DIRECTORY "${OMTALK_LLVM_BINARY_DIR}"
)

message(STATUS "Using CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
find_package(MLIR REQUIRED PATHS "${OMTALK_LLVM_BINARY_DIR}/lib/cmake/mlir" NO_DEFAULT_PATH)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# These must be global for tablegen to work
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${OMTALK_LLVM_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${OMTALK_LLVM_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

message(STATUS "Using MLIRConfig.cmake in:     ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in:     ${LLVM_DIR}")
message(STATUS "Using MLIR_CMAKE_DIR:          ${MLIR_CMAKE_DIR}")
message(STATUS "Using LLLVM_CMAKE_DIR:         ${LLVM_CMAKE_DIR}")
message(STATUS "Using LLVM_INCLUDE_DIRS:       ${LLVM_INCLUDE_DIRS}")
message(STATUS "Using MLIR_INCLUDE_DIRS:       ${MLIR_INCLUDE_DIRS}")
message(STATUS "Using LLVM_BUILD_LIBRARY_DIR:  ${LLVM_BUILD_LIBRARY_DIR}")
message(STATUS "Using LLVM_DEFINITIONS:        ${LLVM_DEFINITIONS}")

###
### LLVM Build Targets
###

# TODO: properly import targets from MLIR. mlir-tblgen is not an exported
# target, and dependencies are not represented. Therefore we have to force a
# build of LLVM at configure time to make sure tblgen is built.
execute_process(
	COMMAND "${CMAKE_COMMAND}" --build .
	WORKING_DIRECTORY "${OMTALK_LLVM_BINARY_DIR}"
)

add_custom_target(omtalk_llvm_project ALL
	COMMAND "${CMAKE_COMMAND}" --build .
	WORKING_DIRECTORY "${OMTALK_LLVM_BINARY_DIR}"
	COMMENT "Building LLVM"
	USES_TERMINAL
)

###
### LLVM Interface
###

add_library(omtalk_llvm INTERFACE)

add_dependencies(omtalk_llvm omtalk_llvm_project)

target_include_directories(omtalk_llvm
    INTERFACE
        ${LLVM_INCLUDE_DIRS}
)

target_compile_definitions(omtalk_llvm
	INTERFACE
		${LLVM_DEFINITIONS}
)

###
### MLIR Interface
###

add_library(omtalk_mlir INTERFACE)

add_dependencies(omtalk_mlir omtalk_llvm_project)

target_include_directories(omtalk_mlir
	INTERFACE
		${LLVM_INCLUDE_DIRS}
		${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(omtalk_mlir
	INTERFACE
		${LLVM_DEFINITIONS}
)
