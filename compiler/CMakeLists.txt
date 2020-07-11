add_mlir_dialect_library(MLIROmtalk
	lib/Dialect/Omtalk/OmtalkDialect.cpp
	lib/Dialect/Omtalk/OmtalkOps.cpp
)

target_include_directories(obj.MLIROmtalk
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${CMAKE_CURRENT_BINARY_DIR}/include
)

target_include_directories(MLIROmtalk
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${CMAKE_CURRENT_BINARY_DIR}/include
)

target_link_libraries(MLIROmtalk
	PUBLIC
		omtalk_mlir
		MLIRIR
)

add_subdirectory(include)
add_subdirectory(lib)
