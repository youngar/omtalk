add_om_library(om-ir
  src/OmDialect.cpp
  src/OmOps.cpp
)

add_dependencies(om-ir om-ir-incgen)

target_include_directories(om-ir
  PUBLIC
    include/
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

target_link_libraries(om-ir
  PUBLIC
    MLIRIR
)

add_subdirectory(include/om/IR)