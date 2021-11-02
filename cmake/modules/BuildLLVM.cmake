if(BUILD_LLVM_)
	return()
endif()
set(BUILD_LLVM_ TRUE)

###
### LLVM CMake Integration
###
if(OM_BUILD_LLVM)
	message(STATUS "Using LLVM_ENABLE_PROJECTS:    ${LLVM_ENABLE_PROJECTS}")
	message(STATUS "Using OM_LLVM_OPTIONS:     ${OM_LLVM_OPTIONS}")

	set(OM_LLVM_SOURCE_DIR "${omtalk_SOURCE_DIR}/external/llvm-project")
	set(OM_LLVM_BINARY_DIR "${omtalk_BINARY_DIR}/external/llvm-project")

	file(MAKE_DIRECTORY "${OM_LLVM_BINARY_DIR}")
	set(extra_find_args NO_DEFAULT_PATH PATHS "${OM_LLVM_BINARY_DIR}")

	if(NOT OMTALK_RAN_LLVM_CMAKE)
	message(STATUS "Building LLVM")
		execute_process(
			COMMAND "${CMAKE_COMMAND}" "${OM_LLVM_SOURCE_DIR}/llvm"
			-G "${CMAKE_GENERATOR}"
			${OM_LLVM_OPTIONS}
			"-DLLVM_ENABLE_PROJECTS=${LLVM_ENABLE_PROJECTS}"
			WORKING_DIRECTORY "${OM_LLVM_BINARY_DIR}"
		)

		# TODO: properly import targets from MLIR. mlir-tblgen is not an exported
		# target, and dependencies are not represented. Therefore we have to force a
		# build of LLVM at configure time to make sure tblgen is built.
		execute_process(
			COMMAND "${CMAKE_COMMAND}" --build .
			WORKING_DIRECTORY "${OM_LLVM_BINARY_DIR}"
		)

		set(OMTALK_RAN_LLVM_CMAKE ON CACHE INTERNAL "")
		list(APPEND extra_find_args "PATHS" "${OM_LLVM_BINARY_DIR}")
	endif()

endif()

find_package(MLIR REQUIRED ${extra_find_args})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# These must be global for tablegen to work
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${OM_LLVM_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${OM_LLVM_BINARY_DIR}/lib)
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

# TODO: I don't think there is any benefit to adding a custom target for llvm
# add_custom_target(omtalk_llvm_project ALL
# 	COMMAND "${CMAKE_COMMAND}" --build .
# 	WORKING_DIRECTORY "${OM_LLVM_BINARY_DIR}"
# 	COMMENT "Building LLVM"
# 	USES_TERMINAL
# )

###
### LLVM Interface
###

add_library(omtalk_llvm INTERFACE)

# add_dependencies(omtalk_llvm omtalk_llvm_project)

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
