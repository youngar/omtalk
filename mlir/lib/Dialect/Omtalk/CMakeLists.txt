add_mlir_dialect_library(MLIROmtalk
    OmtalkDialect.cpp
    OmtalkOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/mlir/include

    DEPENDS
    MLIROmtalkOpsIncGen
    )

target_link_libraries(MLIROmtalk PUBLIC MLIRIR)
