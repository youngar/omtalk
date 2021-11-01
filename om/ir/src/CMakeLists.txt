add_mlir_dialect_library(MLIROm
	OmDialect.cpp
	OmOps.cpp

  DEPENDS
  MLIROmIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

target_include_directories(obj.MLIROm
	PUBLIC
    ${PROJECT_SOURCE_DIR}/compiler/include
    ${PROJECT_BINARY_DIR}/compiler/include
)

target_include_directories(MLIROm
	PUBLIC
    ${PROJECT_SOURCE_DIR}/compiler/include
    ${PROJECT_BINARY_DIR}/compiler/include
)
