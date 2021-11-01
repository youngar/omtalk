add_mlir_dialect_library(MLIROmtalk
	OmtalkDialect.cpp
	OmtalkOps.cpp

  DEPENDS
  MLIROmtalkIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

target_include_directories(obj.MLIROmtalk
	PUBLIC
    ${PROJECT_SOURCE_DIR}/compiler/include
    ${PROJECT_BINARY_DIR}/compiler/include
)

target_include_directories(MLIROmtalk
	PUBLIC
    ${PROJECT_SOURCE_DIR}/compiler/include
    ${PROJECT_BINARY_DIR}/compiler/include
)