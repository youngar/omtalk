### LLVM

set(LLVM_PROJECT_SOURCE_DIRECTORY ${PROJECT_SOURCE_DIR}/external/llvm-project)
set(LLVM_PROJECT_BINARY_DIRECTORY ${PROJECT_BINARY_DIR}/external/llvm-project)

set(LLVM_INCLUDES 
	${LLVM_PROJECT_SOURCE_DIRECTORY}/llvm/include
	${LLVM_PROJECT_BINARY_DIRECTORY}/llvm/include)

list(APPEND CMAKE_MODULE_PATH ${LLVM_PROJECT_SOURCE_DIRECTORY}/llvm/cmake/modules)

### MLIR

set(TABLEGEN_OUTPUT "")
set(LLVM_COMMON_DEPENDS "")

set(MLIR_TABLEGEN_EXE mlir-tblgen)
set(MLIR_SOURCE_DIRECTORY ${LLVM_PROJECT_SOURCE_DIRECTORY}/mlir)
set(MLIR_BINARY_DIRECTORY ${LLVM_PROJECT_BINARY_DIRECTORY}/llvm/tools/mlir)
set(MLIR_INCLUDES ${MLIR_SOURCE_DIRECTORY}/include ${MLIR_BINARY_DIRECTORY}/include)

function(omtalk_tablegen)
	tablegen(MLIR ${ARGV} "-I${MLIR_SOURCE_DIRECTORY}/include" "-I${MLIR_BINARY_DIRECTORY}/include")
	set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
		PARENT_SCOPE)
endfunction(omtalk_tablegen)

### libomtalk-mlir

add_library(libomtalk-mlir
	src/Conversion/OmtalkToLLVM.cpp
	src/Dialect/Omtalk/OmtalkDialect.cpp
	src/Dialect/Omtalk/Ops.cpp
	src/Transforms/Canonicalize.cpp
	src/Transforms/DeadCodeElimination.cpp
)

target_include_directories(libomtalk-mlir
	PUBLIC
		include
		${CMAKE_CURRENT_BINARY_DIR}/include/
		${CMAKE_CURRENT_BINARY_DIR}/src/
		${MLIR_INCLUDES}
		${LLVM_INCLUDES}
)

add_dependencies(libomtalk-mlir omtalk-canonicalize-tblgen)
add_dependencies(libomtalk-mlir omtalk-mlir-tblgen)

target_link_libraries(libomtalk-mlir
	PUBLIC 
		MLIRAffineOps
		MLIRAffineToStandard
		MLIRAnalysis
		MLIRExecutionEngine
		MLIRIR
		MLIRLLVMIR
		MLIRLoopToStandard
		MLIRParser
		MLIRPass
		MLIRStandardOps
		MLIRStandardToLLVM
		MLIRTargetLLVMIR
		MLIRTransforms
)

# llvm libs must be linked afterwordsS
llvm_map_components_to_libnames(llvm_libs support core irreader)
target_link_libraries(libomtalk-mlir
	PUBLIC 
		${llvm_libs}
)

### omtalk-mlir

add_executable(omtalk-mlir
		src/main.cpp
)

# llvm libs must be linked afterwordsS
target_link_libraries(omtalk-mlir
	PRIVATE 
		libomtalk-mlir
)

add_subdirectory(include/mlir/Dialect/Omtalk)
add_subdirectory(src)
add_subdirectory(test)